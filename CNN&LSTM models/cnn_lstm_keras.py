import os
import numpy as np
import tensorflow as tf
from numpy import newaxis
from sklearn import datasets, linear_model
from sklearn.svm import SVR
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from rmv_seas import remove_trend

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Convolution
kernel_size_1 = 8
filters_1 = 64
pool_size_1 = 2

# LSTM
lstm_output_size = 70

#Dense
num_units_dense = 1

def build_model():

    model = Sequential()
    model.add(Conv1D(filters_1,
                     kernel_size_1,
                     padding='valid',
                     activation='relu',
                     strides=1,
                     input_shape=(None,1)))

    model.add(MaxPooling1D(pool_size=pool_size_1))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(num_units_dense))
    model.add(Activation('linear'))

    model.compile(loss="mse", optimizer="rmsprop")

    return model

def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect
    # only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def generate_lags(data, num_lags, bootstrap_size):
    assert bootstrap_size > num_lags

    series = data['time series'].values
    removed_seasonality = data['removed seasonality'].values
    removed_std = data['removed std'].values

    # generate lags
    X = []
    for i in range(bootstrap_size, len(series)):
        X.append(series[i - num_lags:i])
    X = np.array(X)

    y = series[bootstrap_size:]
    removed_seasonality = removed_seasonality[bootstrap_size:]
    removed_std = removed_std[bootstrap_size:]
    assert X.shape[0] == y.shape[0]

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, removed_seasonality, removed_std


def run_epochs(X, y, removed_seasonality, removed_std):

    epochs = 30

    bootstrap_size = 2 * 24 * 25
    n_test = 2 * 24 * 5
    preds_lstm = []
    preds_lr = []
    preds_svr = []
    trues = []

    count = 0
    for i in range(bootstrap_size, len(y), n_test):
        print "Current window:", i, i + n_test

        X_train = X[:i, :]
        y_train = y[:i]
        X_test = X[i:i + n_test, :]
        y_test = y[i:i + n_test]
        trues.append(
            y_test * removed_std[i:i + n_test] + removed_seasonality[i:i + n_test])

        # support vector regression
        svr = SVR()
        svr.fit(X_train[:, :, 0], y_train)
        preds_svr.append(svr.predict(
            X_test[:, :, 0]) * removed_std[i:i + n_test] + removed_seasonality[i:i + n_test])

        # linear regression
        regr = linear_model.LinearRegression()
        regr.fit(X_train[:, :, 0], y_train)
        preds_lr.append(regr.predict(
            X_test[:, :, 0]) * removed_std[i:i + n_test] + removed_seasonality[i:i + n_test])

        # LSTM
        model = build_model()
        model.fit(
            X_train,
            y_train,
            batch_size=128,
            epochs=epochs,
            validation_split=0.00,
            verbose=0)
        """
        print model.layers[0].input.shape
        print model.layers[0].output.shape
        print model.layers[1].output.shape
        print model.layers[2].output.shape
        print model.layers[3].output.shape
        print model.layers[4].output.shape
        """

        # one step ahead prediction
        predicted = predict_point_by_point(model, X_test)

        preds_lstm.append(
            predicted * removed_std[i:i + n_test] + removed_seasonality[i:i + n_test])

        mae = np.mean(
            np.abs(np.array(preds_lstm[count]) - np.array(trues[count])))
        rmse = np.sqrt(
            np.mean((np.array(preds_lstm[count]) - np.array(trues[count]))**2))
        print "CNN_LSTM:    %.3f   %.3f" % (mae, rmse)

        count += 1


def main():


    df = remove_trend(16,27)
    X, y, removed_seasonality, removed_std = generate_lags(df, 20, 2*24)
    run_epochs(X, y, removed_seasonality, removed_std)


if __name__ == "__main__":
    main()
