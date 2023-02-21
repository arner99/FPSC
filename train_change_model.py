from models.utils import *
from models.train import train_model

for iteration in range(4):
    fp = f"data/models/change/input/transitions-{iteration}.tsv"
    X, y = generate_change_train_data(fp)
    train_model(X, y, loss="binary_crossentropy", output_activation="sigmoid", train_epochs=2, dropout=0.0)