import numpy as np
from network.ipa_feature_table import IPAFeatureTable

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


if __name__ == "__main__":
    generate_train_data(-7)
