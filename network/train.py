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


def train_model(X, y, **kwargs):
    # unpack parameters. default values match our model architecture.
    save_dir = kwargs["save_dir"] if "save_dir" in kwargs else "./models"
    model_name = kwargs["model_name"] if "model_name" in kwargs else "model"
    width = kwargs["width"] if "width" in kwargs else 128
    depth = kwargs["depth"] if "depth" in kwargs else 3
    input_dim = kwargs["input_dim"] if "input_dim" in kwargs else 34
    activation = kwargs["activation"] if "activation" in kwargs else "gelu"
    dropout = kwargs["dropout"] if "dropout" in kwargs else 0.05
    early_stopping = kwargs["early_stopping"] if "early_stopping" in kwargs else None
    train_epochs = kwargs["train_epochs"] if "train_epochs" in kwargs else 200
    plot = kwargs["plot"] if "plot" in kwargs else False
    verbose = kwargs["verbose"] if "verbose" in kwargs else False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    if activation == "prelu":
        if depth == 3:
            model = Sequential([
                Dense(width, input_dim=input_dim),
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
                Dense(width, input_dim=input_dim),
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
                Dense(width, input_dim=input_dim),
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
                Dense(width, input_dim=input_dim),
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
                Dense(width, input_dim=input_dim, activation=activation),
                Dropout(dropout),
                Dense(width, activation=activation),
                Dropout(dropout),
                Dense(width, activation=activation),
                Dense(1)
            ])
        elif depth == 2:
            model = Sequential([
                Dense(width, input_dim=input_dim, activation=activation),
                Dropout(dropout),
                Dense(width, activation=activation),
                Dense(1)
            ])
        elif depth == 1:
            model = Sequential([
                Dense(1, activation=activation, input_dim=input_dim)
            ])
        else:
            raise ValueError

    model_fp = "%s/%s.hdf5" % (save_dir, model_name)
    checkpointer = ModelCheckpoint(model_fp, save_best_only=True)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    if early_stopping is None:
        history = model.fit(X_train, y_train, epochs=train_epochs, validation_split=0.1, callbacks=[checkpointer])
    else:
        es = EarlyStopping(patience=early_stopping)
        history = model.fit(X_train, y_train, epochs=train_epochs, validation_split=0.1, callbacks=[checkpointer, es])

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
    X, y, _ = load_train_data("data/train-data.txt")
    train_model(X, y, verbose=True, plot=True)
