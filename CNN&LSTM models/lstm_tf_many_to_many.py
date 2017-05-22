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

learning_rate = 0.001

num_layers = 1
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
target_placeholder = tf.placeholder(tf.float32, [None, seq_len])
Hin_placeholder = tf.placeholder(tf.float32, [None, state_size * num_layers])
keep_prob_placeholder = tf.placeholder(tf.float32)

""" Mini batch generator """
def data_batch_generator(X, batch_size, seq_len):
    startidx = np.random.randint(0, len(X) - seq_len - 1, size = batch_size)

    while True:
        items = np.array([X[start:start + seq_len + 1] for start in startidx])
        startidx = (startidx + seq_len) % (len(X) - seq_len - 1)
        yield items

def prep_batch_for_network(batch):
    x_seq = np.zeros((len(batch), seq_len, num_features), dtype='float32')
    y_seq = np.zeros((len(batch), seq_len), dtype='int32')

    for i, item in enumerate(batch):
        for j in range(seq_len):
            x_seq[i, j] = item[j]
            y_seq[i, j] = item[j + 1]

    return x_seq, y_seq

def lags_generator(series, removed_seasonality, removed_std):
  """ Test lags
  """
  bootstrap_size = 2 * 24

  X = []
  y = []
  rmv_sea= []
  rmv_std = []
  for i in range(bootstrap_size, len(series)):
    
    X.append(series[i - seq_len:i])
    y.append(series[i - seq_len +1 :i + 1])
    rmv_sea.append(removed_seasonality[i - seq_len +1 :i + 1])
    rmv_std.append(removed_std[i - seq_len +1 :i + 1])

  X = np.array(X)
  y = np.array(y)
  rmv_sea = np.array(rmv_sea)
  rmv_std = np.array(rmv_std)

  X = X.reshape(X.shape[0], X.shape[1], 1)
  y = y.reshape(y.shape[0], y.shape[1])
  rmv_sea = rmv_sea.reshape(rmv_sea.shape[0], rmv_sea.shape[1])
  rmv_std = rmv_std.reshape(rmv_std.shape[0], rmv_std.shape[1])

  return X, y, rmv_sea, rmv_std

""" Build the graph """

with tf.name_scope("LSTM") as scope:
   onecell = tf.contrib.rnn.GRUCell(state_size)
   onecell = tf.contrib.rnn.DropoutWrapper(onecell, output_keep_prob=keep_prob_placeholder)
   multicell = tf.contrib.rnn.MultiRNNCell([onecell] * num_layers, state_is_tuple=False)
   outputs, H = tf.nn.dynamic_rnn(multicell, input_placeholder, dtype = tf.float32, initial_state=Hin_placeholder)
   last_output = outputs[:, outputs.shape[1] - 1, :]
   #all_outputs = tf.reshape(outputs, [-1, seq_len * state_size])

with tf.name_scope("Fully_Connected1") as scope:

   W_fc1 = tf.Variable(tf.truncated_normal(
            [state_size, seq_len], stddev=0.01))
   b_fc1 = tf.Variable(tf.constant(0.1, shape=[seq_len]),name = 'b_fc1')
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

    for i in range(bootstrap_size, len(X) , n_test):
      print("Current window", i, i + n_test)

      sess.run(tf.global_variables_initializer())

      X_train = X[:i]

      gen_train = data_batch_generator(X_train, batch_size, seq_len)
      istate = np.zeros([batch_size, state_size*num_layers])  # initial zero input state

      for epoch in range(num_epochs):
        for num_batch in range(len(X_train) / batch_size):

          batch_X_train, batch_y_train = prep_batch_for_network(next(gen_train))

          _, outH = sess.run([opt, H], feed_dict={ input_placeholder: batch_X_train,
                                        target_placeholder: batch_y_train,
                                        Hin_placeholder: istate,
                                        keep_prob_placeholder: keep_prob_train
                                       })
          istate = outH

      X_test, y_test, rmv_sea, rmv_std = lags_generator(X[i:i + n_test], 
                                                       removed_seasonality[i:i + n_test],
                                                       removed_std[i:i + n_test])

      vali_nullstate = np.zeros([X_test.shape[0], state_size*num_layers])

      preds = sess.run(pred, feed_dict= { input_placeholder: X_test,
                                          target_placeholder: y_test,
                                          Hin_placeholder: vali_nullstate,
                                          keep_prob_placeholder: keep_prob_test
                                        })

      preds = preds * rmv_std + rmv_sea
      trues = y_test * rmv_std + rmv_sea

      stacked_preds.extend(preds)
      stacked_ground_truth.extend(trues)

    rmse = np.sqrt(np.mean((np.array(stacked_preds) - np.array(stacked_ground_truth))**2))
    print rmse
    rmses.append(rmse)
    
  errors = pd.DataFrame()
  errors['rmse'] = rmses
  errors.to_csv("data/lstm_many_to_many_errors.csv")
