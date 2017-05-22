import numpy as np
import tensorflow as tf
import pandas as pd
from rmv_seas import remove_trend

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Convolution
kernel_size_1 = 8
filters_1 = 64
pool_size_1 = 2
num_channels = 1

kernel_size_2 = 8
filters_2 = 32
pool_size_2 = 2

# LSTM
state_size = 70
num_layers = 2
shift = 1
keep_prob_test = 1
keep_prob_train = 0.2

# Dense 
num_units = 1

learning_rate = 0.01

# Global variables
num_epochs = 30
batch_size = 128
seq_len = 48

bootstrap_size = 2 * 24 * 25
n_test = 2 * 24 * 5

""" Placeholders """
input_placeholder = tf.placeholder(tf.float32, [None, seq_len, 1])
target_placeholder = tf.placeholder(tf.float32, [None, 1])
Hin_placeholder = tf.placeholder(tf.float32, [None, state_size * num_layers])
keep_prob_placeholder = tf.placeholder(tf.float32)
bn_train = tf.placeholder(tf.bool)

""" Mini batch generator """

def batch_train_generator(X, batch_size, seq_len):

  startidx = np.random.randint(0, len(X) - seq_len, batch_size)
  while True:
    batch_X = np.array([X[start:start + seq_len] for start in startidx])
    batch_y = np.array([X[start:start + seq_len + shift] for start in startidx])
            
    batch_y = batch_y[:, -1]
    startidx = (startidx + seq_len) % (len(X) - seq_len)
    yield batch_X.reshape(batch_size, seq_len, 1), batch_y.reshape(batch_size, 1)

def lags_generator(series, removed_seasonality, removed_std):
  """ Test lags
  """
  bootstrap_size = 2 * 24

  X = []
  for i in range(bootstrap_size, len(series)):
    X.append(series[i - seq_len:i])
  X = np.array(X)

  y = series[bootstrap_size:]
  removed_seasonality = removed_seasonality[bootstrap_size:]
  removed_std = removed_std[bootstrap_size:]
  assert X.shape[0] == y.shape[0]

  X = X.reshape(X.shape[0], X.shape[1], 1)

  return X, y, removed_seasonality, removed_std

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

""" Build the graph """

with tf.name_scope("Conv1") as scope:
  W_conv1 = tf.Variable(tf.truncated_normal(shape=[kernel_size_1 , num_channels, filters_1], stddev=0.01))
  b_conv1 = tf.Variable(tf.constant(0.1, shape=[filters_1]), name = 'bias_for_Conv_Layer_1')
  a_conv1 = tf.nn.conv1d(input_placeholder, W_conv1, stride = 1, padding = "VALID") + b_conv1

with tf.name_scope('Batch_norm_conv1') as scope:
  a_conv1_bn = batch_norm(a_conv1, filters_1, bn_train)
  h_conv1 = tf.nn.relu(a_conv1_bn)

with tf.name_scope("Max_Pooling1") as scope:
  a_pool1 = tf.layers.max_pooling1d(h_conv1, pool_size = pool_size_1, strides = 1)

with tf.name_scope("Conv2") as scope:
  W_conv2 = tf.Variable(tf.truncated_normal(shape=[kernel_size_2 , filters_1, filters_2], stddev=0.01))
  b_conv2 = tf.Variable(tf.constant(0.1, shape=[filters_2]), name = 'bias_for_Conv_Layer_2')
  a_conv2 = tf.nn.conv1d(a_pool1, W_conv2, stride = 1, padding = "VALID") + b_conv2

with tf.name_scope('Batch_norm_conv2') as scope:
  a_conv2_bn = batch_norm(a_conv2, filters_2, bn_train)
  h_conv2 = tf.nn.relu(a_conv2_bn)

with tf.name_scope("Max_Pooling2") as scope:
  a_pool2 = tf.layers.max_pooling1d(h_conv2, pool_size = pool_size_2, strides = 1)

with tf.name_scope("LSTM") as scope:
  onecell = tf.contrib.rnn.GRUCell(state_size)
  onecell = tf.contrib.rnn.DropoutWrapper(onecell, output_keep_prob=keep_prob_placeholder)
  multicell = tf.contrib.rnn.MultiRNNCell([onecell] * num_layers, state_is_tuple=False)
  outputs, H = tf.nn.dynamic_rnn(multicell, a_pool2, dtype = tf.float32)
  last_output = outputs[:, outputs.shape[1] - 1, :]
  #all_outputs = tf.reshape(outputs, [-1, seq_len * state_size])

with tf.name_scope("Fully_Connected1") as scope:
  W_fc1 = tf.Variable(tf.truncated_normal(shape=[state_size, num_units], stddev=0.01))
  b_fc1 = tf.Variable(tf.constant(0.1, shape=[num_units]),name = 'b_fc1')
  pred = tf.matmul(last_output, W_fc1) + b_fc1

with tf.name_scope("Cost") as scope:
  mse = tf.reduce_mean(tf.pow(tf.subtract(pred, target_placeholder), 2.0))

with tf.name_scope("Optimize") as scope:
  optimizer = tf.train.RMSPropOptimizer(learning_rate)
  opt = optimizer.minimize(mse)

""" Run graph """

with tf.Session() as sess:

  rmses = []

  for dest in [26,27,28]:

    df = remove_trend(16,dest)

    X = df['time series'].values
    removed_seasonality = df['removed seasonality'].values
    removed_std = df['removed std'].values

    stacked_preds = []
    stacked_ground_truth = []

    for i in range(bootstrap_size, len(X), n_test):
      print("Current window", i, i + n_test)

      sess.run(tf.global_variables_initializer())

      X_train = X[:i]

      gen_train = batch_train_generator(X_train, batch_size, seq_len)
      istate = np.zeros([batch_size, state_size*num_layers])  # initial zero input state

      for epoch in range(num_epochs):
        for num_batch in range(len(X_train) / batch_size):

          batch_X_train, batch_y_train = next(gen_train)

          _, outH = sess.run([opt, H], feed_dict={ input_placeholder: batch_X_train,
                                        target_placeholder: batch_y_train,
                                        Hin_placeholder: istate,
                                        keep_prob_placeholder: keep_prob_train,
                                        bn_train: True
                                       })
          istate = outH

      X_test, y_test, rmv_sea, rmv_std = lags_generator(X[i:i + n_test], 
                                                       removed_seasonality[i:i + n_test],
                                                       removed_std[i:i + n_test])

      vali_nullstate = np.zeros([X_test.shape[0], state_size*num_layers])


      preds = sess.run(pred, feed_dict= { input_placeholder: X_test,
                                          target_placeholder: y_test.reshape(y_test.shape[0],1),
                                          Hin_placeholder: vali_nullstate,
                                          keep_prob_placeholder: keep_prob_test,
                                          bn_train: False
                                        })

      preds = preds[:, 0] * rmv_std + rmv_sea
      trues = y_test * rmv_std + rmv_sea
        
      stacked_preds.extend(preds)
      stacked_ground_truth.extend(trues)

    rmse = np.sqrt(np.mean((np.array(stacked_preds) - np.array(stacked_ground_truth))**2))
    rmses.append(rmse)

  errors = pd.DataFrame()
  errors['rmse'] = rmses
  errors.to_csv("data/cnn_+_lstm_errors.csv")
        

