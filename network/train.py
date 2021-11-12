import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, PReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from io_utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from itertools import product


def train_model(X, y, save_dir, width, depth, activation, dropout,
                early_stopping=None, plot=False, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    if activation == "prelu":
        if depth == 3:
            model = Sequential([
                Dense(width, input_dim=34),
                PReLU(),
                Dropout(dropout),
                Dense(width),
                PReLU(),
                Dropout(dropout),
                Dense(width),
                PReLU(),
                Dense(1)
            ])
        elif depth == 2:
            model = Sequential([
                Dense(width, input_dim=34),
                PReLU(),
                Dropout(dropout),
                Dense(width),
                PReLU(),
                Dense(1)
            ])
        else:
            raise ValueError
    elif activation == "leaky-relu":
        if depth == 3:
            model = Sequential([
                Dense(width, input_dim=34),
                LeakyReLU(),
                Dropout(dropout),
                Dense(width),
                LeakyReLU(),
                Dropout(dropout),
                Dense(width),
                LeakyReLU(),
                Dense(1)
            ])
        elif depth == 2:
            model = Sequential([
                Dense(width, input_dim=34),
                LeakyReLU(),
                Dropout(dropout),
                LeakyReLU(width),
                PReLU(),
                Dense(1)
            ])
        else:
            raise ValueError
    else:
        if depth == 3:
            model = Sequential([
                Dense(width, input_dim=34, activation=activation),
                Dropout(dropout),
                Dense(width, activation=activation),
                Dropout(dropout),
                Dense(width, activation=activation),
                Dense(1)
            ])
        elif depth == 2:
            model = Sequential([
                Dense(width, input_dim=34, activation=activation),
                Dropout(dropout),
                Dense(width, activation=activation),
                Dense(1)
            ])
        elif depth == 1:
            model = Sequential([
                Dense(1, activation=activation, input_dim=34)
            ])
        else:
            raise ValueError

    model_fp = "%s/model.hdf5" % save_dir
    checkpointer = ModelCheckpoint(model_fp, save_best_only=True)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    if early_stopping is None:
        history = model.fit(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[checkpointer])
    else:
        es = EarlyStopping(patience=early_stopping)
        history = model.fit(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[checkpointer, es])

    val_losses = history.history["val_loss"]
    best_epoch = val_losses.index(min(val_losses)) + 1

    model.load_weights(model_fp)

    y_pred = model.predict(X)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    if verbose:
        print("TRAIN MSE: %f" % train_mse)
        print("TEST MSE: %f" % test_mse)
        print("BEST EPOCH: %i" % best_epoch)

    if plot:
        plt.scatter(y_pred_train, y_train)
        plt.title("Train")
        plt.show()

        plt.scatter(y_pred_test, y_test)
        plt.title("Test")
        plt.show()

        plt.scatter(y_pred, y)
        plt.title("All")
        plt.show()

    return best_epoch, train_mse, test_mse


if __name__ == "__main__":
    X, y, _ = load_train_data("neural_approaches/refined/blank-train-data.txt")
    train_model(X, y, "models", 128, 3, "gelu", 0.05, verbose=True, plot=True)
