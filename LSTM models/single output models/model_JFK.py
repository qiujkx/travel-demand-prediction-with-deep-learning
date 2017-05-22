import functools
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from sklearn import linear_model
from sklearn.svm import SVR
import pandas as pd
from rmv_seas import remove_trend

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
            tf.float32, [None, self.config.seq_len, 1])
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

    def batch_train_generator(self, X, batch_size, seq_len):
        """Consecutive mini
        batch generator
        """
        startidx = np.random.randint(0, len(X) - seq_len, batch_size)
        while True:
            batch_X = np.array([X[start:start + seq_len]
                                for start in startidx])
            batch_y = np.array(
                [X[start:start + seq_len + self.config.shift] for start in startidx])
            batch_y = batch_y[:, -1]
            startidx = (startidx + seq_len) % (len(X) - seq_len)
            yield batch_X.reshape(batch_size, seq_len, 1), batch_y.reshape(batch_size, 1)

    def lags_generator(self, series, removed_seasonality, removed_std):
        """ Test lags
        """
        bootstrap_size = 2 * 24

        X = []
        for i in range(bootstrap_size, len(series)):
            X.append(series[i - self.config.seq_len:i])
        X = np.array(X)

        y = series[bootstrap_size:]
        removed_seasonality = removed_seasonality[bootstrap_size:]
        removed_std = removed_std[bootstrap_size:]
        assert X.shape[0] == y.shape[0]

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, y, removed_seasonality, removed_std

    def evaluate(self, preds, trues):

        rmse = np.sqrt(np.mean((np.array(preds) - np.array(trues))**2))
        return rmse

    def run_epochs(self, X, removed_seasonality, removed_std):
        """
        Train network with sliding window strategy
        """
        bootstrap_size_train = 2 * 24 * 25
        n_test = 2 * 24 * 5

        # arrays for storing 5 day step errors
        lstm_errors = []
        lr_errors = []
        svr_errors = []

        # arrays for stacking predictions
        preds_lstm = []
        preds_lr = []
        preds_svr = []
        ground_truth = []

        sess = tf.Session()

        for i in range(bootstrap_size_train, len(X), n_test):
            print("Current window:", i, i + n_test)

            X_train = X[:i]
            X_test = X[i:i + n_test]
            removed_seasonality_ = removed_seasonality[i:i + n_test]
            removed_std_ = removed_std[i:i + n_test]

            sess.run(tf.global_variables_initializer())

            # mini batch generator for training
            gen_train = self.batch_train_generator(
                X_train, self.config.batch_size, self.config.seq_len)
            train_state = np.zeros(
                [self.config.batch_size, self.config.state_size * self.config.num_layers])

            # all training data in 1 batch for validation
            batch_size = len(X_train) // self.config.seq_len
            gen_val = self.batch_train_generator(
                X_train, batch_size, self.config.seq_len)
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
                X_train, removed_seasonality_, removed_std_)
            x_test, y_test, removed_seasonality_, removed_std_ = self.lags_generator(
                X_test, removed_seasonality_, removed_std_)
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

            # linear regression

            regr = linear_model.LinearRegression()
            regr.fit(x_train[:, :, 0], y_train)
            preds = regr.predict(
                x_test[:, :, 0]) * removed_std_ + removed_seasonality_

            preds_lr.extend(preds)

            lr_errors.append(self.evaluate(preds, trues))

            # support vector regression

            svr = SVR()
            svr.fit(x_train[:, :, 0], y_train)
            preds = svr.predict(
                x_test[:, :, 0]) * removed_std_ + removed_seasonality_

            preds_svr.extend(preds)

            svr_errors.append(self.evaluate(preds, trues))


        # compute overall error metrics

        LSTM_RMSE = self.evaluate(preds_lstm, ground_truth)
        LReg_RMSE = self.evaluate(preds_lr, ground_truth)
        SVR_RMSE = self.evaluate(preds_svr, ground_truth)

        OV_RMSE = pd.DataFrame()

        OV_RMSE['LSTM_RMSE'] = [LSTM_RMSE]
        OV_RMSE['LReg_RMSE'] = [LReg_RMSE]
        OV_RMSE['SVR_RMSE'] = [SVR_RMSE]

        OV_RMSE.to_csv('data/OV_JFK_SO.csv', index=False, sep="\t")

        # save errors computed at each 5 day step

        RMSE = pd.DataFrame()

        RMSE['lstm'] = lstm_errors
        RMSE['lr'] = lr_errors
        RMSE['svr'] = svr_errors

        RMSE.to_csv('data/JFK_SO.csv', index=False, sep="\t")


def main():


    df = remove_trend(16, 26)

    config = Config()
    model = RNN_NeuralModel(config)
    model.run_epochs(df['time series'].values,
                         df['removed seasonality'].values,
                         df['removed std'].values)


if __name__ == "__main__":
    main()
