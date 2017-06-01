
import tensorflow as tf
import functools
import numpy as np
import pandas as pd
from tabulate import tabulate


# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


################################################

# LSTM 1


class Config_LSTM_1:

    def __init__(self):

        self.batch_size = 120
        self.seq_len = 20
        self.state_size = 120
        self.num_layers = 2
        self.dropout_train = 0.25
        self.dropout_eval = 1


class LSTM_Model_1:

    def __init__(self, config):

        self.config = config
        self.add_placeholders()
        self.get_input_placeholder
        self.get_dropout_placeholder
        self.add_LSTM_layer

    def add_placeholders(self):

        with tf.variable_scope("lstm_placeholders_model_1"):

            self.input_placeholder = tf.placeholder(
                tf.float32, [None, self.config.seq_len, 1])
            self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_input_placeholder(self):
        return self.input_placeholder

    def get_dropout_placeholder(self):
        return self.dropout_placeholder

    def add_LSTM_layer(self):

        with tf.variable_scope("lstm_layer_model_1"):

            onecell = tf.contrib.rnn.GRUCell(self.config.state_size)
            onecell = tf.contrib.rnn.DropoutWrapper(
                onecell, output_keep_prob=self.dropout_placeholder)
            multicell = tf.contrib.rnn.MultiRNNCell(
                [onecell] * self.config.num_layers, state_is_tuple=False)
            outputs, _ = tf.nn.dynamic_rnn(
                multicell, self.input_placeholder, dtype=tf.float32)
            return outputs

######################################################

# LSTM 2


class Config_LSTM_2:

    def __init__(self):

        self.batch_size = 128
        self.seq_len = 20
        self.state_size = 150
        self.num_layers = 1
        self.dropout_train = 0.25
        self.dropout_eval = 1


class LSTM_Model_2:

    def __init__(self, config):

        self.config = config
        self.add_placeholders()
        self.get_input_placeholder
        self.get_dropout_placeholder
        self.add_LSTM_layer

    def add_placeholders(self):

        with tf.variable_scope("lstm_placeholders_model_2"):

            self.input_placeholder = tf.placeholder(
                tf.float32, [None, self.config.seq_len, 1])
            self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_input_placeholder(self):
        return self.input_placeholder

    def get_dropout_placeholder(self):
        return self.dropout_placeholder

    def add_LSTM_layer(self):

        with tf.variable_scope("lstm_layer_model_2"):

            onecell = tf.contrib.rnn.GRUCell(self.config.state_size)
            onecell = tf.contrib.rnn.DropoutWrapper(
                onecell, output_keep_prob=self.dropout_placeholder)
            multicell = tf.contrib.rnn.MultiRNNCell(
                [onecell] * self.config.num_layers, state_is_tuple=False)
            outputs, _ = tf.nn.dynamic_rnn(
                multicell, self.input_placeholder, dtype=tf.float32)
            return outputs

########################################################

# LSTM 3


class Config_LSTM_3:

    def __init__(self):

        self.batch_size = 256
        self.seq_len = 20
        self.state_size = 120
        self.num_layers = 1
        self.dropout_train = 0.25
        self.dropout_eval = 1


class LSTM_Model_3:

    def __init__(self, config):

        self.config = config
        self.add_placeholders()
        self.get_input_placeholder
        self.get_dropout_placeholder
        self.add_LSTM_layer

    def add_placeholders(self):

        with tf.variable_scope("lstm_placeholders_model_3"):

            self.input_placeholder = tf.placeholder(
                tf.float32, [None, self.config.seq_len, 1])
            self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_input_placeholder(self):
        return self.input_placeholder

    def get_dropout_placeholder(self):
        return self.dropout_placeholder

    def add_LSTM_layer(self):

        with tf.variable_scope("lstm_layer_model_3"):

            onecell = tf.contrib.rnn.GRUCell(self.config.state_size)
            onecell = tf.contrib.rnn.DropoutWrapper(
                onecell, output_keep_prob=self.dropout_placeholder)
            multicell = tf.contrib.rnn.MultiRNNCell(
                [onecell] * self.config.num_layers, state_is_tuple=False)
            outputs, _ = tf.nn.dynamic_rnn(
                multicell, self.input_placeholder, dtype=tf.float32)
            return outputs

##########################################################


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class Merged_Model:

    def __init__(self):

        self.num_epochs = 10
        self.batch_size = 256
        self.learning_rate = 0.01

        self.add_placeholders()
        self.prediction
        self.add_dense_layer
        self.add_models
        self.cost
        self.optimize

    def add_placeholders(self):
        self.target_placeholder = tf.placeholder(tf.float32, [None, 3])

    @lazy_property
    def cost(self):
        return tf.reduce_mean(tf.pow(tf.subtract(self.prediction, self.target_placeholder), 2.0))

    @lazy_property
    def optimize(self):
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        return optimizer.minimize(self.cost)

    def add_models(self):

        self.config_lstm_1 = Config_LSTM_1()
        self.config_lstm_2 = Config_LSTM_2()
        self.config_lstm_3 = Config_LSTM_3()

        self.model_1 = LSTM_Model_1(self.config_lstm_1)
        self.model_2 = LSTM_Model_2(self.config_lstm_2)
        self.model_3 = LSTM_Model_3(self.config_lstm_3)

        output_1 = self.model_1.add_LSTM_layer()
        output_2 = self.model_2.add_LSTM_layer()
        output_3 = self.model_3.add_LSTM_layer()

        return output_1, output_2, output_3

    def add_dense_layer(self, _input, hidden_size, out_size):

        weight = tf.Variable(tf.truncated_normal(
            [hidden_size, out_size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        return tf.matmul(_input, weight) + bias

    @lazy_property
    def prediction(self):

        output_1, output_2, output_3 = self.add_models()
        merged_output = tf.concat([output_1, output_2, output_3], 2)
        last_output = merged_output[:, merged_output.shape[1] - 1, :]

        prediction = self.add_dense_layer(last_output, int(merged_output.get_shape()[
                                          2]), int(self.target_placeholder.get_shape()[1]))
        return prediction

    def run_epochs(self, X1_train, y1_train, X2_train, y2_train, X3_train, y3_train,
    					X1_test, y1_test, X2_test, y2_test, X3_test, y3_test):

    	y_train = np.column_stack((y1_train, y2_train, y3_train))
    	y_test = np.column_stack((y1_test, y2_test, y3_test))

    	# MULTI-OUTPUT LSTM

    	sess = tf.Session()
    	sess.run(tf.global_variables_initializer())

    	for epoch in range(self.num_epochs):
    		for j in range(len(X1_train) / self.batch_size):

    			batch_X1_train = X1_train[j * self.batch_size:(j + 1) * self.batch_size, :, :]
    			batch_y1_train = y1_train[j * self.batch_size:(j + 1) * self.batch_size]
    			batch_y1_train = batch_y1_train.reshape(len(batch_y1_train), 1)

    			batch_X2_train = X2_train[j * self.batch_size:(j + 1) * self.batch_size, :, :]
    			batch_y2_train = y2_train[j * self.batch_size:(j + 1) * self.batch_size]
    			batch_y2_train = batch_y2_train.reshape(len(batch_y2_train), 1)

    			batch_X3_train = X3_train[j * self.batch_size:(j + 1) * self.batch_size, :, :]
    			batch_y3_train = y3_train[j * self.batch_size:(j + 1) * self.batch_size]
    			batch_y3_train = batch_y3_train.reshape(len(batch_y3_train), 1)

    			batch_y_train = np.concatenate((batch_y1_train, batch_y2_train, batch_y3_train), axis=1)

    			_ = sess.run(self.optimize, feed_dict={self.model_1.get_input_placeholder(): batch_X1_train,
                                                           self.model_2.get_input_placeholder(): batch_X2_train,
                                                           self.model_3.get_input_placeholder(): batch_X3_train,
                                                           self.target_placeholder: batch_y_train,
                                                           self.model_1.get_dropout_placeholder(): self.config_lstm_1.dropout_train,
                                                           self.model_2.get_dropout_placeholder(): self.config_lstm_2.dropout_train,
                                                           self.model_3.get_dropout_placeholder(): self.config_lstm_3.dropout_train
                                                           })

    		error = sess.run(self.cost, feed_dict={self.model_1.get_input_placeholder(): X1_train,
                                                       self.model_2.get_input_placeholder(): X2_train,
                                                       self.model_3.get_input_placeholder(): X3_train,
                                                       self.target_placeholder: y_train,
                                                       self.model_1.get_dropout_placeholder(): self.config_lstm_1.dropout_eval,
                                                       self.model_2.get_dropout_placeholder(): self.config_lstm_2.dropout_eval,
                                                       self.model_3.get_dropout_placeholder(): self.config_lstm_3.dropout_eval
                                                       })
    	print "Epoch: %d, train error: %f" % (epoch, error)

    	preds = sess.run(self.prediction, feed_dict={self.model_1.get_input_placeholder(): X1_test,
                                                         self.model_2.get_input_placeholder(): X2_test,
                                                         self.model_3.get_input_placeholder(): X3_test,
                                                         self.target_placeholder: y_test,
                                                         self.model_1.get_dropout_placeholder(): self.config_lstm_1.dropout_eval,
                                                         self.model_2.get_dropout_placeholder(): self.config_lstm_2.dropout_eval,
                                                         self.model_3.get_dropout_placeholder(): self.config_lstm_3.dropout_eval
                                                         })

        # compute error metrics

        rmse_1 = np.sqrt(np.mean((np.array(preds[:,0]) - np.array(y1_test))**2))
        rmse_2 = np.sqrt(np.mean((np.array(preds[:,1]) - np.array(y2_test))**2))
        rmse_3 = np.sqrt(np.mean((np.array(preds[:,2]) - np.array(y3_test))**2))

        df = pd.DataFrame()

        df["Link_1 RMSE"] = [rmse_1]
        df["Link_2 RMSE"] = [rmse_2]
        df["Link_3 RMSE"] = [rmse_3]

        print tabulate(df, headers='keys', tablefmt='psql')


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

	data = data[(26 <= data.LineDirectionLinkOrder) & (data.LineDirectionLinkOrder <= 28)]

	grouping = data.groupby(['LinkRef'])

	return grouping


def generate_features(data):

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
	#not all the groups have the same number of observations...make them be same size
	mat = mat[0:5000, :]

	X = mat[:, 0:num_lags]
	y = mat[:, num_lags+num_dummies]

	assert X.shape[0] == y.shape[0]

	X = np.reshape(X, (X.shape[0], X.shape[1], 1))

	return X, y


def split_into_train_test(X, y):
    
    # Split data into train and test    
    X_train, X_test = np.split(X, [int(.8*len(X))])
    y_train, y_test = np.split(y, [int(.8*len(y))])

    return X_train, y_train, X_test, y_test


def main():

	l = []
	groups = data_preprocessing()

	for key, group in groups:

		l.append(generate_features(group))

	X1, X2, X3 = l[0][0], l[1][0], l[2][0]
	y1, y2, y3 = l[0][1], l[1][1], l[2][1]

	X1_train, y1_train, X1_test, y1_test = split_into_train_test(X1, y1)
	X2_train, y2_train, X2_test, y2_test = split_into_train_test(X2, y2)
	X3_train, y3_train, X3_test, y3_test = split_into_train_test(X3, y3)

	merged_model = Merged_Model()
	merged_model.run_epochs(X1_train, y1_train,
							X2_train, y2_train,
							X3_train, y3_train,
							X1_test, y1_test,
							X2_test, y2_test,
							X3_test, y3_test)
	

if __name__ == "__main__":
	main()	
