from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def train_model(X, y, **kwargs):
    # unpack parameters. default values match model architecture for the sound correspondence model.
    save_dir = kwargs["save_dir"] if "save_dir" in kwargs else "data/models/corr/trained_models"
    model_name = kwargs["model_name"] if "model_name" in kwargs else "corr-model"
    loss = kwargs["loss"] if "loss" in kwargs else "mean_squared_error"
    test_split = kwargs["test_split"] if "test_split" in kwargs else 0.1
    early_stopping = kwargs["early_stopping"] if "early_stopping" in kwargs else None
    train_epochs = kwargs["train_epochs"] if "train_epochs" in kwargs else 200
    plot = kwargs["plot"] if "plot" in kwargs else False
    verbose = kwargs["verbose"] if "verbose" in kwargs else False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)

    if "input_dim" not in kwargs:
        kwargs["input_dim"] = X.shape[1]  # num of features
    layers = generate_model_layers(**kwargs)

    # compile model
    model = Sequential(layers)
    model.compile(loss=loss, optimizer="adam", metrics=[loss])

    # add checkpointer and, if specified, early stopping
    model_fp = "%s/%s.hdf5" % (save_dir, model_name)
    checkpointer = ModelCheckpoint(model_fp, save_best_only=True)
    if early_stopping is None:
        history = model.fit(X_train, y_train, epochs=train_epochs, validation_split=test_split, callbacks=[checkpointer])
    else:
        es = EarlyStopping(patience=early_stopping)
        history = model.fit(X_train, y_train, epochs=train_epochs, validation_split=test_split, callbacks=[checkpointer, es])

    val_losses = history.history["val_loss"]
    best_epoch = val_losses.index(min(val_losses)) + 1

    # load best model (with the lowest validation loss)
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
        plt.title(f"Train - {model_name}")
        plt.show()

        plt.scatter(y_pred_test, y_test)
        plt.title(f"Test - {model_name}")
        plt.show()

        plt.scatter(y_pred, y)
        plt.title(f"All - {model_name}")
        plt.show()

    return best_epoch, train_mse, test_mse
