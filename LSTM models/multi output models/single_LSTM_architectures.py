import tensorflow as tf

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

        self.batch_size = 128
        self.seq_len = 20
        self.state_size = 150
        self.num_layers = 3
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
