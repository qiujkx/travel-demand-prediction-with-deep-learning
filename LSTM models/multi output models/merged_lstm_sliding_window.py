from single_LSTM_architectures import *
import tensorflow as tf
import functools
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVR

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


class Merged_Model:

    def __init__(self):

        self.num_epochs = 50
        self.batch_size = 128
        self.learning_rate = 0.001

        self.add_placeholders()
        self.prediction
        self.add_dense_layer
        self.add_models
        self.cost
        self.optimize
        self.evaluate

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

    def evaluate(self, preds, trues):
    
        rmse = np.sqrt(np.mean((np.array(preds) - np.array(trues))**2))
        return rmse

    def run_epochs(self, X1, X2, X3, y1, y2, y3,
                   removed_seasonality1, removed_seasonality2, removed_seasonality3,
                   removed_std1, removed_std2, removed_std3):

        bootstrap_size = 2 * 24 * 25
        n_test = 2 * 24 * 5

        # arrays for storing computed performance every 5 days
        lstm_out_1 = []
        lstm_out_2 = []
        lstm_out_3 = []

        lr_out_1 = []
        lr_out_2 = []
        lr_out_3 = []

        svr_out_1 = []
        svr_out_2 = []
        svr_out_3 = []

        # arrays for stacking predictions every 5 days
        # (used for computing overall error metrics)
        preds_lstm_1 = []
        preds_lstm_2 = []
        preds_lstm_3 = []

        preds_lr_1 = []
        preds_lr_2 = []
        preds_lr_3 = []

        preds_svr_1 = []
        preds_svr_2 = []
        preds_svr_3 = []

        ground_truth_1 = []
        ground_truth_2 = []
        ground_truth_3 = []

        sess = tf.Session()

        for i in range(bootstrap_size, len(X1), n_test):
            print("Current window", i, i + n_test)

            X1_train = X1[:i, :]
            y1_train = y1[:i]
            X1_test = X1[i:i + n_test, :]
            y1_test = y1[i:i + n_test]

            X2_train = X2[:i, :]
            y2_train = y2[:i]
            X2_test = X2[i:i + n_test, :]
            y2_test = y2[i:i + n_test]

            X3_train = X3[:i, :]
            y3_train = y3[:i]
            X3_test = X3[i:i + n_test, :]
            y3_test = y3[i:i + n_test]

            y_train = np.column_stack((y1_train, y2_train, y3_train))
            y_test = np.column_stack((y1_test, y2_test, y3_test))

            # MULTI-OUTPUT LSTM

            # re-initialize LSTM
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.num_epochs):
                for j in range(len(X1_train) / self.batch_size):

                    batch_X1_train = X1_train[j *
                                              self.batch_size:(j + 1) * self.batch_size, :, :]
                    batch_y1_train = y1_train[j *
                                              self.batch_size:(j + 1) * self.batch_size]
                    batch_y1_train = batch_y1_train.reshape(
                        len(batch_y1_train), 1)

                    batch_X2_train = X2_train[j *
                                              self.batch_size:(j + 1) * self.batch_size, :, :]
                    batch_y2_train = y2_train[j *
                                              self.batch_size:(j + 1) * self.batch_size]
                    batch_y2_train = batch_y2_train.reshape(
                        len(batch_y2_train), 1)

                    batch_X3_train = X3_train[j *
                                              self.batch_size:(j + 1) * self.batch_size, :, :]
                    batch_y3_train = y3_train[j *
                                              self.batch_size:(j + 1) * self.batch_size]
                    batch_y3_train = batch_y3_train.reshape(
                        len(batch_y3_train), 1)

                    batch_y_train = np.concatenate(
                        (batch_y1_train, batch_y2_train, batch_y3_train), axis=1)

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

            preds_1 = preds[:, 0] * removed_std1[i:i + n_test] + \
                removed_seasonality1[i:i + n_test]
            preds_2 = preds[:, 1] * removed_std2[i:i + n_test] + \
                removed_seasonality2[i:i + n_test]
            preds_3 = preds[:, 2] * removed_std3[i:i + n_test] + \
                removed_seasonality3[i:i + n_test]

            trues_1 = y1_test * \
                removed_std1[i:i + n_test] + removed_seasonality1[i:i + n_test]
            trues_2 = y2_test * \
                removed_std2[i:i + n_test] + removed_seasonality2[i:i + n_test]
            trues_3 = y3_test * \
                removed_std3[i:i + n_test] + removed_seasonality3[i:i + n_test]

            # Stack predictions for computing overall error metrics
            preds_lstm_1.extend(preds_1)
            preds_lstm_2.extend(preds_2)
            preds_lstm_3.extend(preds_3)

            # Stack ground truth for computing overall error metrics
            ground_truth_1.extend(trues_1)
            ground_truth_2.extend(trues_2)
            ground_truth_3.extend(trues_3)

            # Evaluate performance every 5 days
            lstm_out_1.append(self.evaluate(preds_1, trues_1))
            lstm_out_2.append(self.evaluate(preds_2, trues_2))
            lstm_out_3.append(self.evaluate(preds_3, trues_3))

            # MULTI OUTPUT LINEAR REGRESSION

            # generate multivariate input
            X_train = []
            for num_sample in range(X1_train.shape[0]):
                X_train.append(np.concatenate(
                    [X1_train[num_sample, :, 0], X2_train[num_sample, :, 0], X3_train[num_sample, :, 0]]))
            X_train = np.array(X_train)

            X_test = []
            for num_sample in range(X1_test.shape[0]):
                X_test.append(np.concatenate(
                    [X1_test[num_sample, :, 0], X2_test[num_sample, :, 0], X3_test[num_sample, :, 0]]))
            X_test = np.array(X_test)

            # fit independent models
            regr_1 = linear_model.LinearRegression()
            regr_1.fit(X_train, y1_train.reshape(y1_train.shape[0], 1))
            preds_1 = regr_1.predict(
                X_test).reshape(-1) * removed_std1[i:i + n_test] + removed_seasonality1[i:i + n_test]

            regr_2 = linear_model.LinearRegression()
            regr_2.fit(X_train, y2_train.reshape(y2_train.shape[0], 1))
            preds_2 = regr_2.predict(
                X_test).reshape(-1) * removed_std2[i:i + n_test] + removed_seasonality2[i:i + n_test]

            regr_3 = linear_model.LinearRegression()
            regr_3.fit(X_train, y3_train.reshape(y3_train.shape[0], 1))
            preds_3 = regr_3.predict(
                X_test).reshape(-1) * removed_std3[i:i + n_test] + removed_seasonality3[i:i + n_test]

            # Stack predictions for computing overall error metrics
            preds_lr_1.extend(preds_1)
            preds_lr_2.extend(preds_2)
            preds_lr_3.extend(preds_3)

            # Evaluate performance every 5 days
            lr_out_1.append(self.evaluate(preds_1, trues_1))
            lr_out_2.append(self.evaluate(preds_2, trues_2))
            lr_out_3.append(self.evaluate(preds_3, trues_3))
            
            # MULTI OUTPUT SUPPORT VECTOR REGRESSION

            # fit independent models
            svr_1 = SVR()
            svr_1.fit(X_train, y1_train.reshape(y1_train.shape[0], 1))
            preds_1 = svr_1.predict(
                X_test).reshape(-1) * removed_std1[i:i + n_test] + removed_seasonality1[i:i + n_test]

            svr_2 = SVR()
            svr_2.fit(X_train, y2_train.reshape(y2_train.shape[0], 1))
            preds_2 = svr_2.predict(
                X_test).reshape(-1) * removed_std2[i:i + n_test] + removed_seasonality2[i:i + n_test]

            svr_3 = SVR()
            svr_3.fit(X_train, y3_train.reshape(y3_train.shape[0], 1))
            preds_3 = svr_3.predict(
                X_test).reshape(-1) * removed_std3[i:i + n_test] + removed_seasonality3[i:i + n_test]
            
            # Stack predictions for computing overall error metrics
            preds_svr_1.extend(preds_1)
            preds_svr_2.extend(preds_2)
            preds_svr_3.extend(preds_3)

            # Evaluate performance every 5 days
            svr_out_1.append(self.evaluate(preds_1, trues_1))
            svr_out_2.append(self.evaluate(preds_2, trues_2))
            svr_out_3.append(self.evaluate(preds_3, trues_3))


        # compute overall error metrics

        LGA_LSTM_RMSE = self.evaluate(preds_lstm_1, ground_truth_1)
        LGA_LReg_RMSE = self.evaluate(preds_lr_1, ground_truth_1)
        LGA_SVR_RMSE = self.evaluate(preds_svr_1, ground_truth_1)

        JFK_LSTM_RMSE = self.evaluate(preds_lstm_2, ground_truth_2)
        JFK_LReg_RMSE = self.evaluate(preds_lr_2, ground_truth_2)
        JFK_SVR_RMSE = self.evaluate(preds_svr_2, ground_truth_2)

        EWR_LSTM_RMSE = self.evaluate(preds_lstm_3, ground_truth_3)
        EWR_LReg_RMSE = self.evaluate(preds_lr_3, ground_truth_3)
        EWR_SVR_RMSE = self.evaluate(preds_svr_3, ground_truth_3)

        OV_MO_RMSE = pd.DataFrame()

        OV_MO_RMSE['LGA_LSTM_RMSE'] = [LGA_LSTM_RMSE]
        OV_MO_RMSE['LGA_LReg_RMSE'] = [LGA_LReg_RMSE]
        OV_MO_RMSE['LGA_SVR_RMSE'] = [LGA_SVR_RMSE]
        OV_MO_RMSE['JFK_LSTM_RMSE'] = [JFK_LSTM_RMSE]
        OV_MO_RMSE['JFK_LReg_RMSE'] = [JFK_LReg_RMSE]
        OV_MO_RMSE['JFK_SVR_RMSE'] = [JFK_SVR_RMSE]
        OV_MO_RMSE['EWR_LSTM_RMSE'] = [EWR_LSTM_RMSE]
        OV_MO_RMSE['EWR_LReg_RMSE'] = [EWR_LReg_RMSE]
        OV_MO_RMSE['EWR_SVR_RMSE'] = [EWR_SVR_RMSE]

        OV_MO_RMSE.to_csv('data/OV_MO_errors.csv', index=False, sep="\t")

        # save performance errors computed at each 5 day step
        
        LGA_RMSE = pd.DataFrame()

        LGA_RMSE['lstm'] = lstm_out_1
        LGA_RMSE['lr'] = lr_out_1
        LGA_RMSE['svr'] = svr_out_1

        JFK_RMSE = pd.DataFrame()

        JFK_RMSE['lstm'] = lstm_out_2
        JFK_RMSE['lr'] = lr_out_2
        JFK_RMSE['svr'] = svr_out_2

        EWR_RMSE = pd.DataFrame()

        EWR_RMSE['lstm'] = lstm_out_3
        EWR_RMSE['lr'] = lr_out_3
        EWR_RMSE['svr'] = svr_out_3

        JFK_RMSE.to_csv('data/JFK_MO.csv', index=False, sep="\t")
        EWR_RMSE.to_csv('data/EWR_MO.csv', index=False, sep="\t")
        LGA_RMSE.to_csv('data/LGA_MO.csv', index=False, sep="\t")
        

def generate_lags(filename, num_lags, bootstrap_size):
    assert bootstrap_size > num_lags

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

    # generate lags
    X = []
    for i in range(bootstrap_size, len(series)):
        X.append(series[i - num_lags:i])
    X = np.array(X)

    y = series[bootstrap_size:]
    removed_seasonality = removed_seasonality[bootstrap_size:]
    removed_std = removed_std[bootstrap_size:]
    missings = missings[bootstrap_size:]
    assert X.shape[0] == y.shape[0]

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, removed_seasonality, removed_std


def main():

    X1, y1, removed_seasonality1, removed_std1 = generate_lags(
        'data/seasonality_16_27.csv', 20, 2 * 24)
    X2, y2, removed_seasonality2, removed_std2 = generate_lags(
        'data/seasonality_16_26.csv', 20, 2 * 24)
    X3, y3, removed_seasonality3, removed_std3 = generate_lags(
        'data/seasonality_16_28.csv', 20, 2 * 24)

    merged_model = Merged_Model()
    merged_model.run_epochs(X1, X2, X3, y1, y2, y3, removed_seasonality1,
                            removed_seasonality2, removed_seasonality3,
                            removed_std1, removed_std2, removed_std3
                            )


if __name__ == "__main__":
    main()
