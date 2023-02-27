from keras.models import Sequential
import os
import numpy as np
from models.utils import generate_model_layers


def load_model(fp, **kwargs):
    model = Sequential(generate_model_layers(**kwargs))
    model.compile()
    model.load_weights(fp)
    return model


def load_corr_model(fp="data/models/corr/trained_models/corr-model.hdf5", **kwargs):
    if "input_dim" not in kwargs:
        kwargs["input_dim"] = 34
    return load_model(fp, **kwargs)


def load_change_model(fp, **kwargs):
    if "input_dim" not in kwargs:
        kwargs["input_dim"] = 68
    return load_model(fp, **kwargs)


def save_model_weights(model_fp, output_dir, **kwargs):
    # width and depth are hard-coded for our model architecture right now, please modify here when needed
    model = Sequential(generate_model_layers(**kwargs))

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    model.load_weights(model_fp)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(output_dir + "/weights"):
        os.mkdir(output_dir + "/weights")

    for i, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()[0]
        layer_biases = layer.get_weights()[1]
        layer_dir = output_dir + "/weights/layer" + str(i+1)
        os.mkdir(layer_dir)
        np.savetxt("%s/weights.txt" % layer_dir, layer_weights)
        np.savetxt("%s/biases.txt" % layer_dir, layer_biases)


def save_corr_model_weights(model_fp, output_dir, **kwargs):
    if "input_dim" not in kwargs:
        kwargs["input_dim"] = 34
    save_model_weights(model_fp, output_dir, **kwargs)


def save_change_model_weights(model_fp, output_dir, **kwargs):
    if "input_dim" not in kwargs:
        kwargs["input_dim"] = 68
    save_model_weights(model_fp, output_dir, **kwargs)
