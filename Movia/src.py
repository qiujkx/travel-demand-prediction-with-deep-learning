import pandas as pd
import numpy as np
import tensorflow as tf
import functools
from sklearn.svm import SVR
from sklearn import linear_model
from tabulate import tabulate


def data_preprocessing():

	data = pd.read_csv('../data/4A_201701_Consistent.csv', sep=';')

	# Initial data-slicing
	data = data[(data.LinkTravelTime > 0) & (data.LineDirectionCode == 1)]

	# Data convertion
	data['DateTime'] = pd.to_datetime(data['DateTime'])
	time = pd.DatetimeIndex(data['DateTime']) 
	data['TimeOfDayClass'] = 'NO_PEEK' 
	data['Hour'] = time.hour
	data.ix[((7 < time.hour) & (time.hour < 9) & (data['DayType'] == 1)), 'TimeOfDayClass'] = 'PEEK' 
	data.ix[((15 < time.hour) & (time.hour < 17) & (data['DayType'] == 1)), 'TimeOfDayClass'] = 'PEEK' 
	data = data[(27 <= data.LineDirectionLinkOrder) & (data.LineDirectionLinkOrder <= 27)]

	data = data.sort_values("DateTime", ascending=False)
	data = data.set_index(np.arange(0,data.shape[0],1))


	num_lags = 20
	num_dummies = 2
	target = []
	lags = []
	timeOfday = []
	for i in range(data.shape[0]):
	    target.append(data.iloc[i].LinkTravelTime)
	    lags.append(data.LinkTravelTime.shift(-1)[i:i+num_lags].values)
	    timeOfday.append(data.iloc[i].TimeOfDayClass)
	    
	mat = pd.DataFrame(lags)
	mat["timeOfDay"] = timeOfday
	mat = pd.get_dummies(mat)
	mat["ground_truth"] = target
	mat = mat.dropna(axis=0)

	mat = np.array(mat)

	X = mat[:, 0:num_lags]
	y = mat[:, num_lags+num_dummies]

	return X, y


def split_into_train_test(X, y):
    
    # Split data into train and test    
    X_train, X_test = np.split(X, [int(.8*len(X))])
    y_train, y_test = np.split(y, [int(.8*len(y))])

    return X_train, y_train, X_test, y_test


def pred_linear_model(X_train, y_train, X_test, y_test):

	# Train
	clf = linear_model.LinearRegression()
	clf.fit(X_train, y_train) 
	y_train_pred = clf.predict(X_train)
	train_rmse = np.sqrt(np.mean((np.array(y_train_pred) - np.array(y_train))**2))

	# Test
	y_test_pred = clf.predict(X_test)
	test_rmse = np.sqrt(np.mean((np.array(y_test_pred) - np.array(y_test))**2))

	return train_rmse, test_rmse

def pred_SVR(X_train, y_train, X_test, y_test):

	clf = SVR()
	clf.fit(X_train, y_train) 
	y_train_pred = clf.predict(X_train)
	train_rmse = np.sqrt(np.mean((np.array(y_train_pred) - np.array(y_train))**2))
	    
	# Test
	y_test_pred = clf.predict(X_test)
	test_rmse = np.sqrt(np.mean((np.array(y_test_pred) - np.array(y_test))**2))

	return train_rmse, test_rmse

### LSTM ###

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class LstmConfig:

    def __init__(self):

        self.batch_size = 256
        self.seq_len = 20
        self.learning_rate = 0.01
        self.state_size = 128
        self.num_layers = 1
        self.num_epochs = 5
        self.dropout_train = 0.25
        self.dropout_eval = 1

class LstmModel:

    def __init__(self, config):

        self.config = config
        self.add_placeholders()

    def add_placeholders(self):

        self.input_placeholder = tf.placeholder(tf.float32, [None, self.config.seq_len, 1], "input")
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.target_placeholder = tf.placeholder(tf.float32, [None, 1], "target")

    def add_LSTM_layer(self):

        onecell = tf.contrib.rnn.GRUCell(self.config.state_size)
        onecell = tf.contrib.rnn.DropoutWrapper(onecell, output_keep_prob=self.dropout_placeholder)            
        multicell = tf.contrib.rnn.MultiRNNCell([onecell] * self.config.num_layers, state_is_tuple=False)

        outputs, _ = tf.nn.dynamic_rnn(multicell, self.input_placeholder, dtype=tf.float32)
        return outputs
        
    def add_dense_layer(self, input, hidden_size, out_size):

        weight = tf.Variable(tf.truncated_normal([hidden_size, out_size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        return tf.matmul(input, weight) + bias

    @lazy_property
    def cost(self):
        
        mse = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(self.prediction, self.target_placeholder), 2.0)))
        return mse

    @lazy_property
    def optimize(self):
        
        optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.cost)

    def batch_train_generator(self, X, y):
        
        for i in range(len(X) // self.config.batch_size):
            batch_X = X[i:i+self.config.batch_size, :]
            batch_y = y[i:i+self.config.batch_size]

            yield batch_X, batch_y

    def pred_LSTM(self, X_train, y_train, X_test, y_test):

        last_output = self.add_LSTM_layer()
        last_output = last_output[:, last_output.shape[1] - 1, :]
        last_output = self.add_dense_layer(last_output, self.config.state_size, 1)
        self.prediction = last_output
        self.optimize

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(self.config.num_epochs):
            # mini batch generator for training
            gen_train = self.batch_train_generator(X_train, y_train)

            for batch in range(len(X_train) // self.config.batch_size):

                batch_X, batch_y = next(gen_train);
                _ = sess.run(self.optimize, feed_dict={
                        self.input_placeholder: batch_X.reshape(-1, self.config.seq_len, 1),
                        self.target_placeholder: batch_y.reshape(-1, 1),
                        self.dropout_placeholder: self.config.dropout_train
                })

            train_error = sess.run(self.cost, feed_dict={
                    self.input_placeholder: X_train.reshape(-1, self.config.seq_len, 1),
                    self.target_placeholder: y_train.reshape(-1, 1),
                    self.dropout_placeholder: self.config.dropout_eval
            })
            
        
        preds = sess.run(self.prediction, feed_dict={
                self.input_placeholder: X_test.reshape(-1, self.config.seq_len, 1),
                self.dropout_placeholder: self.config.dropout_eval
        })

        test_rmse = np.sqrt(np.mean((np.array(preds) - np.array(y_test))**2))

        return train_error, test_rmse


def main():
    
	X, y = data_preprocessing()
	X_train, y_train, X_test, y_test = split_into_train_test(X, y)

	lr_train_rmse, lr_test_rmse = pred_linear_model(X_train, y_train, X_test, y_test)
	svr_train_rmse, svr_test_rmse = pred_SVR(X_train, y_train, X_test, y_test)

	config = LstmConfig()
	model = LstmModel(config)
	lstm_train_rmse, lstm_test_rmse = model.pred_LSTM(X_train, y_train, X_test, y_test)

	df = pd.DataFrame()

	df["lr_train"] = [lr_train_rmse]
	df["svr_train"] = [svr_train_rmse]
	df["lstm_train"] = [lstm_train_rmse]

	df["lr_test"] = [lr_test_rmse]
	df["svr_test"] = [svr_test_rmse]
	df["lstm_test"] = [lstm_test_rmse]

	print tabulate(df, headers='keys', tablefmt='psql')


if __name__ == "__main__":
	main()