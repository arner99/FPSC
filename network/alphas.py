import matplotlib.pyplot as plt
import numpy as np
import ast
from ipa_feature_table import *
from io_utils import *
from itertools import combinations
from math import log


feature_table = IPAFeatureTable()

# feature vectors as keys
alphas = {}
equivalence_classes = {}

alpha_values = []
non_zero_alpha_values = []

first_line_processed = False
with open("data/cog/transitions.tsv") as f:
    for line in f:
        if line == "":
            continue

        if not first_line_processed:
            symbols = line.strip().split("\t")
            first_line_processed = True
        else:
            fields = line.split()
            symbol = fields[0]

            feat_vec_1 = str(feature_table(symbol))
            if not feat_vec_1:
                continue

            if feat_vec_1 in alphas:
                alphas_for_symbol = alphas.get(feat_vec_1)
            else:
                alphas_for_symbol = {}

            for i, alpha in enumerate(fields[1:]):
                symbol2 = symbols[i]
                feat_vec_2 = str(feature_table(symbol2))

                if not feat_vec_2:
                    continue

                alpha_value = int(alpha)

                if feat_vec_2 in alphas_for_symbol:
                    alphas_for_symbol[feat_vec_2] += alpha_value
                else:
                    alphas_for_symbol[feat_vec_2] = alpha_value

                # record statistics for equivalence classes
                if feat_vec_1 in equivalence_classes:
                    eq_class_1 = equivalence_classes[feat_vec_1]
                    if symbol in eq_class_1:
                        eq_class_1[symbol] += alpha_value
                    else:
                        eq_class_1[symbol] = alpha_value
                else:
                    equivalence_classes[feat_vec_1] = {symbol: alpha_value}

                if feat_vec_2 in equivalence_classes:
                    eq_class_2 = equivalence_classes[feat_vec_2]
                    if symbol2 in eq_class_2:
                        eq_class_2[symbol2] += alpha_value
                    else:
                        eq_class_2[symbol2] = alpha_value
                else:
                    equivalence_classes[feat_vec_2] = {symbol2: alpha_value}

                """
                alpha_values.append(alpha_value)
                if alpha_value != 0:
                    non_zero_alpha_values.append(alpha_value) """

            alphas[feat_vec_1] = alphas_for_symbol


def write_data(mode=None):
    if mode:
        filename = f"data/cog/train-data-{mode}.txt"
    else:
        filename = "data/cog/train-data.txt"

    with open(filename, "w") as f:
        for vec1, alpha_dict in alphas.items():
            feature_vector1 = ast.literal_eval(vec1)
            if not feature_vector1:
                continue
            feat1 = [str(x) for x in feature_vector1]
            # get most frequent symbol from equivalence class
            sym1 = max(equivalence_classes[vec1], key=equivalence_classes[vec1].get)
            freq1 = sum(equivalence_classes[vec1].values())
            for vec2, alpha in alpha_dict.items():
                feature_vector2 = ast.literal_eval(vec2)
                if not feature_vector2:
                    continue
                feat2 = [str(x) for x in feature_vector2]
                sym2 = max(equivalence_classes[vec2], key=equivalence_classes[vec2].get)
                freq2 = sum(equivalence_classes[vec2].values())
                if alpha > 0:
                    if mode == "log":
                        log_alpha = log(alpha)
                        f.write(" ".join([str(log_alpha), sym1, sym2, str(freq1),
                                          str(freq2)] + feat1 + feat2) + "\n")
                    elif mode == "double-log":
                        if alpha <= 1:
                            continue
                        double_log_alpha = log(log(alpha))
                        f.write(" ".join([str(double_log_alpha), sym1, sym2, str(freq1),
                                          str(freq2)] + feat1 + feat2) + "\n")
                    elif mode == "normalized":
                        normalized_alpha = alpha / (freq1 + freq2)
                        f.write(" ".join([str(normalized_alpha), sym1, sym2, str(freq1),
                                          str(freq2)]) + "\n")
                    else:
                        f.write(" ".join([str(alpha), sym1, sym2, str(freq1),
                                      str(freq2)] + feat1 + feat2) + "\n")


write_data()
write_data("log")
write_data("double-log")
write_data("normalized")

"""
_, y, pairs = load_train_data("data/cog/train-data-with-freq.txt")
alpha_ratios = {}
relevant_alphas = {}

for alpha, pair in zip(y, pairs):
    if alpha > 100:
        relevant_alphas[pair] = alpha

print("DONE FILTERING")

for pair, alpha in relevant_alphas.items():
    i, j = pair
    if (j, i) in relevant_alphas:
        inv_alpha = relevant_alphas[(j, i)]
        alpha_ratios[pair] = alpha / inv_alpha

print("DONE RATIO CALCULATION")

sorted_alpha_ratios = {x: alpha_ratios[x] for x in sorted(alpha_ratios, key=alpha_ratios.get, reverse=True)}

print("DONE SORTING")

with open("alpha_ratios.txt", "w") as f:
    for pair, ratio in sorted_alpha_ratios.items():
        i, j = pair
        f.write(f"{i} {j} {ratio}\n")
"""