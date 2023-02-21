from models.train import train_model
from models.utils import generate_corr_train_data, load_train_data_from_file

# generate training data
generate_corr_train_data(-7)

# load generated training data and train the correspondence model
X, y, _ = load_train_data_from_file("data/models/corr/input/train-data.txt")
train_model(X, y, verbose=True)  # training params can be set via **kwargs

# load generated training data and train the gap correspondence model
X, y, _ = load_train_data_from_file("data/models/corr/input/gap-train-data.txt")
train_model(X, y, model_name="gap-corr-model", verbose=True)
