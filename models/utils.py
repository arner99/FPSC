import numpy as np
import pandas as pd
import random
from features.ipa_feature_table import IPAFeatureTable
from keras.layers import LeakyReLU, PReLU, Dense, Dropout


def generate_corr_train_data(conf_score_threshold, fp_in="resources/models/corr/input/pmi_scores.tsv",
                             fp_out="resources/models/corr/input/train-data.txt",
                             fp_out_gap="resources/models/corr/input/gap-train-data.txt",
                             fp_conf="resources/models/corr/input/confidence_scores.tsv"):
    relevant_pairs = []
    with open(fp_conf, "r") as f:
        for line in f:
            fields = line.strip().split()
            if fields[0] == "+" or fields[1] == "+" or fields[0] == "#" or fields[1] == "#":
                continue
            conf_score = float(fields[2])
            if conf_score > conf_score_threshold:
                relevant_pairs.append((fields[0], fields[1]))
                relevant_pairs.append((fields[1], fields[0]))

    ipa_feature_table = IPAFeatureTable()
    with open(fp_in, "r") as f:
        with open(fp_out, "w") as wr:
            with open(fp_out_gap) as wr_gap:
                for line in f:
                    fields = line.strip().split()
                    symbol1 = fields[0]
                    symbol2 = fields[1]
                    score = float(fields[2])
                    if (symbol1, symbol2) in relevant_pairs:
                        try:
                            pair_features = ipa_feature_table.get_pair_encoding(symbol1, symbol2)
                            feature_string = " ".join([str(x) for x in pair_features])
                            if symbol1 == "-" or symbol2 == "-":
                                wr_gap.write("%f %s %s %s\n" % (score, symbol1, symbol2, feature_string))
                            else:
                                wr.write("%f %s %s %s\n" % (score, symbol1, symbol2, feature_string))
                        except KeyError:
                            print(f"WARNING: Could not encode symbol ({symbol1}, {symbol2}).")
                            continue


def generate_change_train_data(transition_fp, compression_rate=10, exclude_symbols=None):
    df = pd.read_csv(transition_fp, sep="\t", header=0, index_col=0)

    if exclude_symbols:
        df = df.drop(columns=exclude_symbols, index=exclude_symbols)

    matrix = df.to_numpy()

    ipa_feature_table = IPAFeatureTable()

    illegal_sounds = [x for x in df.index if not ipa_feature_table(x)]

    alphabet = [x for x in df.index if x not in illegal_sounds]
    illegal_sounds_indices = [list(df.index).index(x) for x in illegal_sounds]

    # delete "illegal" sounds (sounds that can not be encoded
    matrix = np.delete(matrix, illegal_sounds_indices, 0)
    matrix = np.delete(matrix, illegal_sounds_indices, 1)

    # compress transition counts by chosen rate, round to next int, cast back to int
    matrix = np.rint(matrix / compression_rate)
    matrix = matrix.astype(int)

    assert matrix.shape[0] == matrix.shape[1] == len(alphabet)

    size = len(alphabet)
    negative_examples = np.zeros(matrix.shape, dtype=int)

    # generate random transition counts, conditioned on all sounds as source and target sounds respectively
    for i, sym in enumerate(alphabet):
        row = matrix[i]
        col = matrix[:, i]

        # count row-wise entries
        for _ in range(np.sum(row)):
            rand_int = random.choice(range(size))
            negative_examples[i, rand_int] += 1

        # count col-wise entries
        for _ in range(np.sum(col)):
            rand_int = random.choice(range(size))
            negative_examples[rand_int, i] += 1

    # double alphas because each value was considered twice (rows and cols // marginal distributions in both directions)
    num_data_points = np.sum(matrix) * 2 + np.sum(negative_examples)

    X = np.zeros((num_data_points, 68))
    y = np.zeros(num_data_points, dtype=int)

    current_index = 0
    for i, sym1 in enumerate(alphabet):
        for j, sym2 in enumerate(alphabet):
            pos_examples = matrix[i, j]
            # again - count every positive example twice
            for _ in range(pos_examples * 2):
                X[current_index] = ipa_feature_table(sym1) + ipa_feature_table(sym2)
                y[current_index] = 1
                current_index += 1
            neg_examples = negative_examples[i, j]
            for _ in range(neg_examples):
                X[current_index] = ipa_feature_table(sym1) + ipa_feature_table(sym2)
                # y stays 0 at that position
                current_index += 1

    return X, y


def load_train_data_from_file(fp):
    sound_pairs = []
    scores = []
    pair_features = []
    with open(fp, "r") as f:
        for line in f:
            fields = line.strip().split()
            sound_pair = (fields[1], fields[2])
            sound_pairs.append(sound_pair)
            score = float(fields[0])
            scores.append(score)
            features = fields[3:]
            pair_features.append(features)
    scores = np.array(scores, dtype=float)
    pair_features = np.array(pair_features, dtype=int)
    return pair_features, scores, sound_pairs


def generate_model_layers(**kwargs):
    # unpack parameters. default values match model architecture for the sound correspondence model.
    width = kwargs["width"] if "width" in kwargs else 128
    depth = kwargs["depth"] if "depth" in kwargs else 3
    input_dim = kwargs["input_dim"] if "input_dim" in kwargs else 34  # num of features
    activation = kwargs["activation"] if "activation" in kwargs else "gelu"
    output_activation = kwargs["output_activation"] if "output_activation" in kwargs else None
    dropout = kwargs["dropout"] if "dropout" in kwargs else 0.05

    layers = []
    advanced_activations = ["prelu", "leaky-relu"]

    # set up layers according to parameters
    for i in range(depth):
        # PReLU and Leaky ReLU are advanced activations that need to be added as separate layers
        if activation in advanced_activations:
            if i == 0:  # specify input dimension (= length of feature vector) for the first layer
                layers.append(Dense(width, input_dim=input_dim))
            else:
                layers.append(Dense(width))

            if activation == "prelu":
                layers.append(PReLU())
            if activation == "leaky-relu":
                layers.append(LeakyReLU)

        else:
            if i == 0:  # specify input dimension (= length of feature vector) for the first layer
                layers.append(Dense(width, input_dim=input_dim, activation=activation))
            else:
                layers.append(Dense(width, activation=activation))

        # add dropout before hidden layers
        if i < depth-1:
            layers.append(Dropout(dropout))

    # add final layer with according activation function if necessary (e.g. sigmoid for a binary classifier)
    if output_activation:
        layers.append(Dense(1, activation=output_activation))
    else:
        layers.append(Dense(1))

    return layers
