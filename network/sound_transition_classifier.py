import random
import pandas as pd
import numpy as np
from ipa_feature_table import *

df = pd.read_csv("data/cog/transitions.tsv", sep="\t", header=0, index_col=0)

df = df.drop(columns=["Unnamed: 3", "+", "#"], index=["+", "#"])
df = df[df.index.notnull()]

matrix = df.to_numpy()

ipa_feature_table = IPAFeatureTable()

entries = 0
illegal_sounds = [x for x in df.index if not ipa_feature_table(x)]

alphabet = [x for x in df.index if x not in illegal_sounds]
illegal_sounds_indices = [list(df.index).index(x) for x in illegal_sounds]

matrix = np.delete(matrix, illegal_sounds_indices, 0)
matrix = np.delete(matrix, illegal_sounds_indices, 1)

print(np.sum(matrix) - matrix[0,0])

assert matrix.shape[0] == matrix.shape[1] == len(alphabet)

for i, sym1 in enumerate(alphabet):
    row = matrix[i]
    col = matrix[:, i]
    if sym1 == "-":
        row = np.delete(row, i)
        col = np.delete(col, i)

    # count row-wise entries
    negative_samples_row = [0] * len(row)
    for _ in range(np.sum(row)):
        rand_int = random.choice(range(len(row)))
        negative_samples_row[rand_int] += 1

    for alpha, negative in zip(row, negative_samples_row):
        entries += abs(alpha - negative)

    # count col-wise entries
    negative_samples_col = [0] * len(col)
    for _ in range(np.sum(col)):
        rand_int = random.choice(range(len(col)))
        negative_samples_col[rand_int] += 1

    for alpha, negative in zip(col, negative_samples_col):
        entries += abs(alpha - negative)

print(entries)
