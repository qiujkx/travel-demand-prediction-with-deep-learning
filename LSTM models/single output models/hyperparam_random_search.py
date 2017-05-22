import functools
import numpy as np
import datetime
import tensorflow as tf
import pandas as pd
import random
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

    def __init__(self, state_size, batch_size, learning_rate):

        self.batch_size = batch_size
        self.num_epochs = 50
        self.seq_len = 48
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.num_layers = 2
        self.dropout_train = 0.5
        self.dropout_eval = 1
        self.shift = 1
        self.num_features = 1
        self.num_units = 1


class RNN_NeuralModel:

    def __init__(self, config):

        self.config = config
        self.add_placeholders()
        self.add_lstm_layer
        self.add_dense_layer
        self.prediction
        self.cost
        self.optimize

    def add_placeholders(self, name="placeholders"):
        """Generate placeholder variables to represent the input tensors
        """
        with tf.name_scope(name):
            self.input_placeholder = tf.placeholder(
                tf.float32, [None, self.config.seq_len, self.config.num_features])
            self.target_placeholder = tf.placeholder(tf.float32, [None, self.config.num_units])
            self.dropout_placeholder = tf.placeholder(tf.float32)
            self.Hin_placeholder = tf.placeholder(
                tf.float32, [None, self.config.state_size * self.config.num_layers])

    def add_lstm_layer(self, name="GRU_layer"):
        """Add lstm layers
        """
        with tf.name_scope(name):
            onecell = tf.contrib.rnn.GRUCell(self.config.state_size)
            onecell = tf.contrib.rnn.DropoutWrapper(
                onecell, output_keep_prob=self.dropout_placeholder)
            multicell = tf.contrib.rnn.MultiRNNCell(
                [onecell] * self.config.num_layers, state_is_tuple=False)
            output, self.H = tf.nn.dynamic_rnn(
                multicell, self.input_placeholder, dtype=tf.float32, initial_state=self.Hin_placeholder)
            return output

    def add_dense_layer(self, _input, in_size, out_size, name="dense_layer"):
        """Add fully connected layer
        """
        with tf.name_scope(name):
            weight = tf.Variable(tf.truncated_normal(
                [in_size, out_size], stddev=0.01))
            bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
            act = tf.matmul(_input, weight) + bias
            return act

    @lazy_property
    def cost(self, name="loss"):
        """Add loss function
        """
        with tf.name_scope(name):
            mse = tf.reduce_mean(
                tf.pow(tf.subtract(self.prediction, self.target_placeholder), 2.0))
            return mse

    @lazy_property
    def optimize(self, name="train_step"):
        """Sets up the training Ops.
        """
        with tf.name_scope(name):
            optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate)
            return optimizer.minimize(self.cost)

    @lazy_property
    def prediction(self):
        """Make predictions.
        """
        output = self.add_lstm_layer()
        last = output[:, output.shape[1] - 1, :]
        prediction = self.add_dense_layer(
            last, self.config.state_size, self.config.num_units)
        return prediction

    def batch_train_generator(self, X):
        """Consecutive mini
        batch generator
        """
        startidx = np.random.randint(
            0, len(X) - self.config.seq_len, self.config.batch_size)
        while True:
            batch_X = np.array([X[start:start + self.config.seq_len]
                                for start in startidx])
            batch_y = np.array(
                [X[start:start + self.config.seq_len + self.config.shift] for start in startidx])
            batch_y = batch_y[:, -1]
            startidx = (startidx + self.config.seq_len) % (len(X) -
                                                           self.config.seq_len)
            yield batch_X.reshape(self.config.batch_size, self.config.seq_len, 1), batch_y.reshape(self.config.batch_size, 1)

    def run_epochs(self, train_set, validation_set, hparam):
        """
        Train network and validate it. Save results for displaying them
        with TensorBoard.
        """
        gen_train = self.batch_train_generator(train_set)
        gen_validation = self.batch_train_generator(validation_set)

        num_batches_train = train_set.shape[0] // self.config.batch_size
        num_batches_validation = validation_set.shape[0] // self.config.batch_size

        tf.summary.scalar("loss", self.cost)
        merged_summary = tf.summary.merge_all()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            writer_train = tf.summary.FileWriter(LOGDIR_train + hparam)
            writer_valid = tf.summary.FileWriter(LOGDIR_valid + hparam)
            writer_valid.add_graph(sess.graph)

            train_state = np.zeros(
                [self.config.batch_size, self.config.state_size * self.config.num_layers])

            for epoch in range(self.config.num_epochs):
                for batch in range(num_batches_train):

                    batch_X_train, batch_y_train = next(gen_train)

                    feed_dict = {self.input_placeholder: batch_X_train,
                                 self.target_placeholder: batch_y_train,
                                 self.Hin_placeholder: train_state,
                                 self.dropout_placeholder: self.config.dropout_train}

                    _, state = sess.run([self.optimize, self.H], feed_dict)
                    train_state = state

                # EVAL-TRAIN
                cur_loss_train = 0
                train_eval_nullstate = np.zeros(
                    [self.config.batch_size, self.config.state_size * self.config.num_layers])

                for batch in range(num_batches_train):

                    batch_X_train, batch_y_train = next(gen_train)

                    feed_dict = {self.input_placeholder: batch_X_train,
                                 self.target_placeholder: batch_y_train,
                                 self.Hin_placeholder: train_eval_nullstate,
                                 self.dropout_placeholder: self.config.dropout_eval}

                    s = sess.run(merged_summary, feed_dict)
                    writer_train.add_summary(s, epoch * num_batches_train + batch)

                    loss_train = sess.run(self.cost, feed_dict)
                    cur_loss_train += loss_train
                train_error = cur_loss_train / num_batches_train
                print "Epoch: %d, train error: %f" % (epoch, train_error)

                # EVAL-VALIDATION
                cur_loss_val = 0
                validation_eval_nullstate = np.zeros(
                    [self.config.batch_size, self.config.state_size * self.config.num_layers])

                for batch in range(num_batches_validation):

                    batch_X_val, batch_y_val = next(gen_validation)

                    feed_dic = {self.input_placeholder: batch_X_val,
                                self.target_placeholder: batch_y_val,
                                self.Hin_placeholder: validation_eval_nullstate,
                                self.dropout_placeholder: self.config.dropout_eval}

                    s = sess.run(merged_summary, feed_dict)
                    writer_valid.add_summary(s, epoch * num_batches_validation + batch)

                    loss = sess.run(self.cost, feed_dict)
                    cur_loss_val += loss
                valid_error = cur_loss_val / num_batches_validation
                print "Epoch: %d, validation error: %f" % (epoch, valid_error)


class Auxiliary_funcs:

    def __init__(self, hyper_config):
        self.config = hyper_config
        self.roundup10

    def split_data_into_training_validation(self, X):

        n_train = np.arange(0, (len(X) * 5 // 10))
        n_validation = np.arange(len(X) * 5 // 10, len(X))

        return X[n_train], X[n_validation]

    def roundup10(self, n):
        return 10 * ((n + 9) // 10)

    def generate_random_hyperparams(self):

        rdm_hidden_size = self.roundup10(random.randint(
            self.config.min_hidden_size, self.config.max_hidden_size))
        rdm_batch_size = random.randint(
            self.config.min_batch_size, self.config.max_batch_size)
        rdm_learning_rate = random.uniform(
            self.config.min_learning_rate, self.config.max_learning_rate)

        return rdm_hidden_size, rdm_batch_size, rdm_learning_rate

    def make_hparam_string(self, hidden_size, batch_size, learning_rate):
        return ("batch_size_%d,hidden_size_%d,learning_rate_%.0E"
                % (batch_size,
                   hidden_size,
                   learning_rate))


class HyperParam_Ranges:

    def __init__(self):

        self.min_hidden_size = 50
        self.max_hidden_size = 300
        self.min_batch_size = 64
        self.max_batch_size = 512
        self.min_learning_rate = 1E-05
        self.max_learning_rate = 1E-03

def main():

    hyper_config = HyperParam_Ranges()
    fn = Auxiliary_funcs(hyper_config)

    df = remove_trend(16, 27)

    X_train, X_validation = fn.split_data_into_training_validation(df['time series'].values)

    for i in range(50):

        internal_hidden_size, batch_size, learning_rate = fn.generate_random_hyperparams()

        # construct hyperparam string for each combination -> Tensorboard
        hparam = fn.make_hparam_string(
            internal_hidden_size, batch_size, learning_rate)

        # clear the default graph
        tf.reset_default_graph()

        config = Config(internal_hidden_size, batch_size, learning_rate)
        model = RNN_NeuralModel(config)
        model.run_epochs(X_train, X_validation, hparam)


LOGDIR_train = 'tf_rdos_train/'
LOGDIR_valid = 'tf_rdos_valid/'
if __name__ == "__main__":
    main()
