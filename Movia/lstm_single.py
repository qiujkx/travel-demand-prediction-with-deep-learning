import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.preprocessing as pp
import functools
from data_parsing import *


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

        self.batch_size = 64
        self.seq_len = 20
        self.learning_rate = 0.001
        self.state_size = 128
        self.num_layers = 2
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
        
    def add_dense_layer(self, _input, hidden_size, out_size):

        weight = tf.Variable(tf.truncated_normal([hidden_size, out_size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        return tf.matmul(_input, weight) + bias

    @lazy_property
    def cost(self):
        """Add loss function
        """
        mse = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(self.prediction, self.target_placeholder), 2.0)))
        return mse

    @lazy_property
    def optimize(self):
        """Sets up the training Ops.
        """
        optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.cost)

    def batch_train_generator(self, X, y):
        """Consecutive mini
        batch generator
        """
        for i in range(len(X) // self.config.batch_size):
            batch_X = X[i:i+self.config.batch_size, :]
            batch_y = y[i:i+self.config.batch_size]

            yield batch_X, batch_y

    def run_epochs(self, X_train, y_train, X_test):

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

            print train_error
        
        preds = sess.run(self.prediction, feed_dict={
                self.input_placeholder: X_test.reshape(-1, self.config.seq_len, 1),
                self.dropout_placeholder: self.config.dropout_eval
        })

        return preds

        
def main():    

    # Configuration
    group_columns = []
    #categorial_columns = ['LinkRef', 'DayType', 'TimeOfDayClass']
    categorical_columns = []
    meta_columns = ['JourneyRef', 'DateTime', 'LineDirectionLinkOrder', 'LinkName']

    # Load and pre-process data
    data = load_csv('../data/4A_201701_Consistent.csv', group_columns = group_columns, categorical_columns = categorical_columns, meta_columns = meta_columns)

    for group, X, Y, meta in data:
        
        print('Group:', group)

        n_train = np.arange(0, (len(X) * 8 // 10))
        n_test = np.arange(len(X) * 8 // 10, len(X))

        X_train = X[n_train]
        y_train = Y[n_train]

        X_test = X[n_test]
        y_test = Y[n_test]

        config = LstmConfig()
        model = LstmModel(config)

        preds = model.run_epochs(X_train, y_train, X_test)

        rmse = np.sqrt(np.mean((np.array(preds) - np.array(y_test))**2))
        print rmse
        

if __name__ == "__main__": 
    main()
