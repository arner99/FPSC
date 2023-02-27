from models.utils import *
from models.train import train_model

for iteration in range(4):
    fp = f"resources/models/change/input/transitions-{iteration}.tsv"
    X, y = generate_change_train_data(fp)
    train_model(X, y, model_name=f"change-model-{iteration}", save_dir="resources/models/change/trained_models",
                loss="binary_crossentropy", output_activation="sigmoid", train_epochs=2, dropout=0.0)
