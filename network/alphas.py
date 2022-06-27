import matplotlib.pyplot as plt
import numpy as np
from ipa_feature_table import *

feature_table = IPAFeatureTable()

alphas = {}
symbols = []

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
            alphas_for_symbol = {}
            for i, alpha in enumerate(fields[1:]):
                alpha_value = int(alpha)
                alphas_for_symbol[symbols[i]] = alpha_value
                alpha_values.append(alpha_value)
                if alpha_value != 0:
                    non_zero_alpha_values.append(alpha_value)
            alphas[symbol] = alphas_for_symbol


with open("data/cog/train-data.txt", "w") as f:
    for sym1, alpha_dict in alphas.items():
        feature_vector1 = feature_table(sym1)
        if not feature_vector1:
            continue
        feat1 = [str(x) for x in feature_vector1]
        for sym2, alpha in alpha_dict.items():
            feature_vector2 = feature_table(sym2)
            if not feature_vector2:
                continue
            feat2 = [str(x) for x in feature_vector2]
            if alpha > 0:
                f.write(" ".join([str(alpha), sym1, sym2] + feat1 + feat2) + "\n")
