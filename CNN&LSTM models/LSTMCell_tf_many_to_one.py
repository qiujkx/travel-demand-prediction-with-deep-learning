import numpy as np
import tensorflow as tf
import pandas as pd
from rmv_seas import remove_trend

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# LSTM
state_size = 120
num_units = 1

learning_rate = 0.001

num_layers = 2
shift = 1

keep_prob_test = 1
keep_prob_train = 0.2

# Global variables
num_epochs = 50
batch_size = 128
seq_len = 48
num_features = 1

bootstrap_size = 2 * 24 * 25
n_test = 2 * 24 * 5

""" Placeholders """

input_placeholder = tf.placeholder(tf.float32, [None, seq_len, num_features])
target_placeholder = tf.placeholder(tf.float32, [None, num_units])
keep_prob_placeholder = tf.placeholder(tf.float32)
init_state_placeholder = tf.placeholder(tf.float32, [num_layers, 2, None, state_size])

state_per_layer_list = tf.unstack(init_state_placeholder, axis=0)
rnn_tuple_state = tuple(
    [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)

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

""" Build the graph """

with tf.name_scope("LSTM") as scope:
   onecell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple = True)
   onecell = tf.contrib.rnn.DropoutWrapper(onecell, output_keep_prob=keep_prob_placeholder)
   multicell = tf.contrib.rnn.MultiRNNCell([onecell] * num_layers, state_is_tuple = True)
   outputs, H = tf.nn.dynamic_rnn(multicell, input_placeholder, dtype = tf.float32, initial_state = rnn_tuple_state)
   last_output = outputs[:, outputs.shape[1] - 1, :]
   #all_outputs = tf.reshape(outputs, [-1, seq_len * state_size])

with tf.name_scope("Fully_Connected1") as scope:

   W_fc1 = tf.Variable(tf.truncated_normal(
            [state_size, num_units], stddev=0.01))
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

  for dest in [26, 27, 28]:

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
      current_state = np.zeros((num_layers, 2, batch_size, state_size))

      for epoch in range(num_epochs):
        for num_batch in range(len(X_train) / batch_size):

          batch_X_train, batch_y_train = next(gen_train)

          _, current_state = sess.run([opt, H], feed_dict={ input_placeholder: batch_X_train,
                                        target_placeholder: batch_y_train,
                                        init_state_placeholder: current_state,
                                        keep_prob_placeholder: keep_prob_train
                                       })

      X_test, y_test, rmv_sea, rmv_std = lags_generator(X[i:i + n_test], 
                                                       removed_seasonality[i:i + n_test],
                                                       removed_std[i:i + n_test])

      vali_nullstate = np.zeros((num_layers, 2, X_test.shape[0], state_size))

      preds = sess.run(pred, feed_dict= { input_placeholder: X_test,
                                          target_placeholder: y_test.reshape(y_test.shape[0],1),
                                          init_state_placeholder: vali_nullstate,
                                          keep_prob_placeholder: keep_prob_test
                                        })

      preds = preds[:, 0] * rmv_std + rmv_sea
      trues = y_test * rmv_std + rmv_sea

      stacked_preds.extend(preds)
      stacked_ground_truth.extend(trues)
        
    rmse = np.sqrt(np.mean((np.array(stacked_preds) - np.array(stacked_ground_truth))**2))
    print rmse
    rmses.append(rmse)
  
  errors = pd.DataFrame()
  errors['rmse'] = rmses
  errors.to_csv("data/lstm_many_to_one_errors.csv")

