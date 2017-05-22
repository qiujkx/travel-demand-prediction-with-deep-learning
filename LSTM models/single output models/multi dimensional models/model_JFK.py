import functools
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from sklearn import linear_model
from sklearn.svm import SVR
import pandas as pd

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class Config:
    """Holds model hyperparams.
    The config class is used to store various hyperparameters.
    """

    def __init__(self):

        self.batch_size = 128
        self.num_epochs = 50
        self.seq_len = 20
        self.learning_rate = 0.001
        self.state_size = 150
        self.num_layers = 1
        self.dropout_train = 0.25
        self.dropout_eval = 1
        self.shift = 1
        self.num_dim = 2


class RNN_NeuralModel:

    def __init__(self, config):

        self.config = config
        self.add_placeholders()
        self.add_lstm_layer
        self.add_dense_layer
        self.prediction
        self.cost
        self.optimize
        self.evaluate

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors
        """
        self.input_placeholder = tf.placeholder(
            tf.float32, [None, self.config.seq_len, 2])
        self.target_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.Hin_placeholder = tf.placeholder(
            tf.float32, [None, self.config.state_size * self.config.num_layers])

    def add_lstm_layer(self):
        """Add lstm layers
        """
        onecell = tf.contrib.rnn.GRUCell(self.config.state_size)
        onecell = tf.contrib.rnn.DropoutWrapper(
            onecell, output_keep_prob=self.dropout_placeholder)
        multicell = tf.contrib.rnn.MultiRNNCell(
            [onecell] * self.config.num_layers, state_is_tuple=False)
        output, self.H = tf.nn.dynamic_rnn(
            multicell, self.input_placeholder, dtype=tf.float32, initial_state=self.Hin_placeholder)
        return output

    def add_dense_layer(self, _input, in_size, out_size):
        """Add fully connected layer
        """
        weight = tf.Variable(tf.truncated_normal(
            [in_size, out_size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        return tf.matmul(_input, weight) + bias

    @lazy_property
    def cost(self):
        """Add loss function
        """
        mse = tf.reduce_mean(
            tf.pow(tf.subtract(self.prediction, self.target_placeholder), 2.0))
        return mse

    @lazy_property
    def optimize(self):
        """Sets up the training Ops.
        """
        optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def prediction(self):
        """Make predictions.
        """
        output = self.add_lstm_layer()
        last = output[:, output.shape[1] - 1, :]
        prediction = self.add_dense_layer(
            last, self.config.state_size, int(self.target_placeholder.get_shape()[1]))
        return prediction

    def batch_train_generator(self, X, W, batch_size, seq_len, num_dim):
        """Consecutive mini
        batch generator
        """
        batch_X = np.zeros((batch_size, seq_len, num_dim))
        startidx = np.random.randint(0, len(X) - seq_len, batch_size)

        while True:
            batch_T = np.array([X[start:start + seq_len]
                                for start in startidx])

            batch_W = np.array([W[start:start + seq_len]
                                for start in startidx])

            for num_batch in range(batch_size):
                batch_X[num_batch, :, 0] = batch_T[num_batch, :]
                batch_X[num_batch, :, 1] = batch_W[num_batch, :]

            batch_y = np.array(
                [X[start:start + seq_len + self.config.shift]
                 for start in startidx])

            batch_y = batch_y[:, -1]
            batch_y = batch_y.reshape(batch_size, 1)

            startidx = (startidx + seq_len) % (len(X) - seq_len)
            yield batch_X, batch_y

    def lags_generator(self, x, w, removed_seasonality, removed_std):
        """ Test lags
        """
        bootstrap_size = 2 * 24

        X = []
        W = []
        for i in range(bootstrap_size, len(x)):
            X.append(x[i - self.config.seq_len:i])
            W.append(w[i - self.config.seq_len:i])
        X = np.array(X)
        W = np.array(W)

        batch_X = np.zeros((X.shape[0], self.config.seq_len, 2))
        for num_batch in range(X.shape[0]):

            batch_X[num_batch, :, 0] = X[num_batch, :]
            batch_X[num_batch, :, 1] = W[num_batch, :]

        batch_y = np.array(x[bootstrap_size:])
        removed_seasonality = removed_seasonality[bootstrap_size:]
        removed_std = removed_std[bootstrap_size:]
        assert batch_X.shape[0] == batch_y.shape[0]

        return batch_X, batch_y, removed_seasonality, removed_std

    def evaluate(self, preds, trues):

        rmse = np.sqrt(np.mean((np.array(preds) - np.array(trues))**2))
        return rmse

    def run_epochs(self, X, W, removed_seasonality, removed_std):
        """
        Train network with sliding window strategy
        """
        bootstrap_size_train = 2 * 24 * 25
        n_test = 2 * 24 * 5

        # arrays for storing 5 day step errors
        lstm_errors = []

        # arrays for stacking predictions
        preds_lstm = []
        ground_truth = []

        sess = tf.Session()

        for i in range(bootstrap_size_train, len(X), n_test):
            print("Current window:", i, i + n_test)

            X_train = X[:i]
            W_train = W[:i]
            X_test = X[i:i + n_test]
            W_test = W[i:i + n_test]
            removed_seasonality_ = removed_seasonality[i:i + n_test]
            removed_std_ = removed_std[i:i + n_test]

            sess.run(tf.global_variables_initializer())

            # mini batch generator for training
            gen_train = self.batch_train_generator(
                X_train, W_train, self.config.batch_size, self.config.seq_len, self.config.num_dim)

            train_state = np.zeros(
                [self.config.batch_size, self.config.state_size * self.config.num_layers])

            # all training data in 1 batch for validation
            batch_size = len(X_train) // self.config.seq_len
            gen_val = self.batch_train_generator(
                X_train, W_train, batch_size, self.config.seq_len, self.config.num_dim)
            x_train, y_train = next(gen_val)

            eval_state = np.zeros(
                [batch_size, self.config.state_size * self.config.num_layers])

            # deep neural network

            for epoch in range(self.config.num_epochs):
                for num_batch in range(len(X_train) / self.config.batch_size):

                    batch_X_train, batch_y_train = next(gen_train)
                    _, outH = sess.run([self.optimize, self.H], feed_dict={self.input_placeholder: batch_X_train,
                                                                           self.target_placeholder: batch_y_train,
                                                                           self.Hin_placeholder: train_state,
                                                                           self.dropout_placeholder: self.config.dropout_train
                                                                           })
                    train_state = outH
                error = sess.run(self.cost, feed_dict={self.input_placeholder: x_train,
                                                       self.target_placeholder: y_train,
                                                       self.Hin_placeholder: eval_state,
                                                       self.dropout_placeholder: self.config.dropout_eval
                                                       })

            print "Epoch: %d, train error: %f" % (epoch, error)

            x_train, y_train, _, _ = self.lags_generator(
                X_train, W_train, removed_seasonality_, removed_std_)
            x_test, y_test, removed_seasonality_, removed_std_ = self.lags_generator(
                X_test, W_test, removed_seasonality_, removed_std_)
            test_state = np.zeros(
                [x_test.shape[0], self.config.state_size * self.config.num_layers])

            preds = sess.run(self.prediction, feed_dict={self.input_placeholder: x_test,
                                                         self.target_placeholder: y_test.reshape(y_test.shape[0], 1),
                                                         self.Hin_placeholder: test_state,
                                                         self.dropout_placeholder: self.config.dropout_eval
                                                         })
            preds = preds[:, 0] * removed_std_ + removed_seasonality_
            trues = y_test * removed_std_ + removed_seasonality_

            preds_lstm.extend(preds)
            ground_truth.extend(trues)

            lstm_errors.append(self.evaluate(preds, trues))

        # compute overall error metrics

        LSTM_RMSE = self.evaluate(preds_lstm, ground_truth)

        OV_RMSE = pd.DataFrame()

        OV_RMSE['LSTM_RMSE'] = [LSTM_RMSE]

        OV_RMSE.to_csv('data/OV_JFK_SO.csv', index=False, sep="\t")

        # save errors computed at each 5 day step

        RMSE = pd.DataFrame()

        RMSE['lstm'] = lstm_errors

        RMSE.to_csv('data/JFK_SO.csv', index=False, sep="\t")


def load_time_series_data(filename):

    # read data from file
    f = open(filename)
    series = []
    removed_seasonality = []
    removed_std = []
    missings = []
    for line in f:
        splt = line.split(",")
        series.append(float(splt[0]))
        removed_seasonality.append(float(splt[1]))
        removed_std.append(float(splt[2]))
    series = np.array(series)
    removed_seasonality = np.array(removed_seasonality)
    removed_std = np.array(removed_std)
    f.close()

    return series, removed_seasonality, removed_std


def load_weather_data(filename):

    weather_data = pd.read_csv(filename, parse_dates=[
                               'date'], index_col='date')

    from_date = datetime.datetime.strptime(
        "2015-07-01 00:00", "%Y-%m-%d %H:%M")
    to_date = datetime.datetime.strptime("2015-12-31 23:30", "%Y-%m-%d %H:%M")

    return weather_data.loc[from_date:to_date].values


def main():

    X, removed_seasonality, removed_std = load_time_series_data(
        'data/seasonality_16_26.csv')

    W = load_weather_data(
        'data/30min_freq_weather_NYC_2015.csv')

    W = W.reshape(-1)

    config = Config()
    model = RNN_NeuralModel(config)
    model.run_epochs(X, W, removed_seasonality, removed_std)


if __name__ == "__main__":
    main()
