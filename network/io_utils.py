import numpy as np
from network.ipa_feature_table import IPAFeatureTable
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, PReLU
import os

features = "syl,son,cons,cont,delrel,lat,nas,strid,voi,sg,cg,ant,cor,distr,lab,hi,lo,back,round,velaric,tense,long,hitone,hireg,backshift,frontshift,opening,centering,closing,longdistance,secondrounded,rising,falling,contour".split(
    ",")


def generate_train_data(conf_score_threshold, fp_in="data/global-corr-model-output.tsv", fp_out="data/train-data.txt",
                        fp_out_blank="data/blank-train-data.txt", fp_conf="data/cldf-global-iw-corr-conf.tsv"):
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

    ipa_feature_table = IPAFeatureTable("data/all_ipa_symbols.csv")
    with open(fp_in, "r") as f:
        with open(fp_out, "w") as wr:
            with open(fp_out_blank) as wr_blank:
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
                                wr_blank.write("%f %s %s %s\n" % (score, symbol1, symbol2, feature_string))
                            else:
                                wr.write("%f %s %s %s\n" % (score, symbol1, symbol2, feature_string))
                        except KeyError:
                            continue


def load_train_data(fp):
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


def index_to_feature(index):
    return features[index]


def feature_to_index(feature):
    return features.index(feature)


def get_features():
    return features


def save_model_weights(model_fp, output_dir):
    # width and depth are hard-coded for our model architecture right now, please modify here when needed
    model = Sequential([
        Dense(128, input_dim=34, activation="gelu"),
        #Dropout(0.05),
        Dense(128, activation="gelu"),
        #Dropout(0.05),
        Dense(128, activation="gelu"),
        Dense(1)
    ])

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


if __name__ == "__main__":
    generate_train_data(-7)
    #save_model_weights("models/model.hdf5", "./models/general_model")
