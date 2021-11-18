import pandas as pd
from itertools import permutations


vowel_dimensions = {
    "i": [1, 1, 0],
    "y": [1, 1, 1],
    "ɨ": [2, 1, 0],
    "ʉ": [2, 1, 1],
    "ɯ": [3, 1, 0],
    "u": [3, 1, 1],
    "ɪ": [1, 1.5, 0],
    "ʏ": [1, 1.5, 1],
    "ʊ": [3, 1.5, 1],
    "e": [1, 2, 0],
    "ø": [1, 2, 1],
    "ɘ": [2, 2, 0],
    "ɵ": [2, 2, 1],
    "ɤ": [3, 2, 0],
    "o": [3, 2, 1],
    "ə": [2, 2.5, 0],
    "ɛ": [1, 3, 0],
    "œ": [1, 3, 1],
    "ɜ": [2, 3, 0],
    "ɞ": [2, 3, 1],
    "ʌ": [3, 3, 0],
    "ɔ": [3, 3, 1],
    "ɐ": [2, 4, 0],
    "æ": [1, 3.5, 0],
    "a": [1, 4, 0],
    "ɶ": [1, 4, 1],
    "ɑ": [3, 4, 0],
    "ɒ": [3, 4, 1]
}

vowels = list(vowel_dimensions.keys())


def add_diphthongs(df, include_long=True, include_nasal=True):
    """
    add feature representations for diphthongs
    :param df:
    :param include_long: whether to also generate (partially) long diphthongs
    :param include_nasal: whether to also generate nasal diphthongs
    :return:
    """
    new_features = ["backshift", "frontshift", "opening", "centering", "closing", "longdistance", "secondrounded"]

    for feature in new_features:
        df[feature] = "0"

    for i, j in permutations(vowels, 2):
        str_repr = i + j
        x_i = vowel_dimensions[i][0]  # horizontal position (front-back axis)
        y_i = vowel_dimensions[i][1]  # vertical position (open-close axis)
        z_i = vowel_dimensions[i][2]  # roundedness
        x_j = vowel_dimensions[j][0]
        y_j = vowel_dimensions[j][1]
        z_j = vowel_dimensions[j][2]
        row = df[df.index == i].iloc[0, :]

        # backshift?
        if x_j > x_i:
            row["backshift"] = "+"
        else:
            row["backshift"] = "-"

        # frontshift?
        if x_j < x_i:
            row["frontshift"] = "+"
        else:
            row["frontshift"] = "-"

        # opening?
        if y_j > y_i and abs(y_j - y_i) >= 1:
            row["opening"] = "+"
        else:
            row["opening"] = "-"

        # closing?
        if y_j < y_i and abs(y_j - y_i) >= 1:
            row["closing"] = "+"
        else:
            row["closing"] = "-"

        # secondrounded?
        if z_j == 1:
            row["secondrounded"] = "+"
        else:
            row["secondrounded"] = "-"

        # longdistance?
        if 2 <= y_i <= 3:
            if abs(y_i - y_j) == 2:
                row["longdistance"] = "+"
            else:
                row["longdistance"] = "-"
        else:
            if abs(y_i - y_j) > 2:
                row["longdistance"] = "+"
            else:
                row["longdistance"] = "-"

        # centering?
        if ((x_j == 2 and 2 <= y_j <= 3) or (y_j == 1.5 and y_i != 1.5) or (y_j == 3.5 and y_i != 3.5)
                or (y_i < 2 and y_j == 2) or (y_i > 3 and y_j == 3)):
            row["centering"] = "+"
        else:
            row["centering"] = "-"

        df = df.append(row)

        tmp = df.index.values.tolist()
        tmp[-1] = str_repr
        df.index = tmp

        if include_nasal:
            row["nas"] = "+"
            str_repr = i + "̃" + j + "̃"  # add tilde on top of vowels

            df = df.append(row)

            tmp = df.index.values.tolist()
            tmp[-1] = str_repr
            df.index = tmp

        if include_long:
            row["nas"] = "-"
            row["long"] = "+"
            len_symbol = "ː"

            for _ in range(3):
                df = df.append(row)

            tmp = df.index.values.tolist()
            tmp[-3] = i + len_symbol + j
            tmp[-2] = i + j + len_symbol
            tmp[-1] = i + len_symbol + j + len_symbol
            df.index = tmp

    return df


def _has_horizontal_bow(triphthong):
    x_vow_1 = vowel_dimensions[triphthong[0]][0]
    x_vow_2 = vowel_dimensions[triphthong[1]][0]
    x_vow_3 = vowel_dimensions[triphthong[2]][0]

    return ((x_vow_2 < x_vow_1) and (x_vow_2 < x_vow_3)) or (x_vow_2 > x_vow_1) and (x_vow_2 > x_vow_3)


def _has_vertical_bow(triphthong):
    y_vow_1 = vowel_dimensions[triphthong[0]][1]
    y_vow_2 = vowel_dimensions[triphthong[1]][1]
    y_vow_3 = vowel_dimensions[triphthong[2]][1]

    return ((y_vow_2 < y_vow_1) and (y_vow_2 < y_vow_3)) or (y_vow_2 > y_vow_1) and (y_vow_2 > y_vow_3)


def _has_bow_trajectory(triphthong):
    return _has_horizontal_bow(triphthong) or _has_vertical_bow(triphthong)


def add_triphthongs(df, include_long=True, include_nasal=True):
    """
    add feature representation for triphthongs
    :param df:
    :param include_long: whether to also generate (partially) long triphthongs
    :param include_nasal: whether to also generate nasal triphthongs
    :return:
    """
    for i, j in permutations(vowels, 2):
        base_diphthong = i + j
        for k in vowels:
            triphthong = i + j + k
            if _has_bow_trajectory(triphthong):
                row = df[df.index == base_diphthong]
                if _has_horizontal_bow(triphthong):
                    row["backshift"] = "+"
                    row["frontshift"] = "+"
                if _has_vertical_bow(triphthong):
                    row["opening"] = "+"
                    row["closing"] = "+"
            else:
                equivalent_diphthong = i + k
                row = df[df.index == equivalent_diphthong]

            df = df.append(row)

            tmp = df.index.values.tolist()
            tmp[-1] = triphthong
            df.index = tmp

            if include_nasal:
                row["nas"] = "+"
                str_repr = i + "̃" + j + "̃" + k + "̃"  # add tilde on top of vowels

                df = df.append(row)

                tmp = df.index.values.tolist()
                tmp[-1] = str_repr
                df.index = tmp

            if include_long:
                row["nas"] = "-"
                row["long"] = "+"
                len_symbol = "ː"

                for _ in range(7):
                    df = df.append(row)

                tmp = df.index.values.tolist()
                tmp[-7] = i + len_symbol + j + k
                tmp[-6] = i + j + len_symbol + k
                tmp[-5] = i + j + k + len_symbol
                tmp[-4] = i + len_symbol + j + len_symbol + k
                tmp[-3] = i + len_symbol + j + k + len_symbol
                tmp[-2] = i + j + len_symbol + k + len_symbol
                tmp[-1] = i + len_symbol + j + len_symbol + k + len_symbol
                df.index = tmp

    return df


def remove_bow(df):
    """
    add representations of polyphthongs and affricates without the tying bow
    :param df: the dataframe to be manipulated
    :return: the expanded dataframe
    """
    df_slice = df[(df.index.str.contains("͡"))].copy()
    df_slice.index = df_slice.index.str.replace("͡", "")
    return pd.concat([df, df_slice])


def add_new_symbols(df):
    """
    add the manually added symbols which are not represented in panphon
    :param df:
    :return:
    """
    new_symbols_df = pd.read_csv("panphon_data/raw/manually_added_symbols.csv", index_col="ipa")
    return pd.concat([df, new_symbols_df])


def include_equivalent_symbols(df):
    """
    includes equivalent symbol representations
    (mainly characters that look the same but have a different unicode representation)
    :param df:
    :return:
    """
    eq_symbols = {}

    with open("panphon_data/raw/equivalent_symbols.csv", "r") as f:
        for row in f:
            fields = row.strip().split(",")
            eq_symbols[fields[0]] = fields[1]

    eq_symbols_df = pd.DataFrame(columns=df.columns.tolist())
    to_be_replaced = eq_symbols.keys()

    for symbol, row in df.iterrows():
        contains_targets = [symbol.__contains__(char) for char in to_be_replaced]
        if any(contains_targets):
            eq_symbols_df = eq_symbols_df.append(row)
            tmp = eq_symbols_df.index.values.tolist()
            for key in eq_symbols:
                tmp[-1] = tmp[-1].replace(key, eq_symbols[key])
            eq_symbols_df.index = tmp

    return pd.concat([df, eq_symbols_df])


if __name__ == "__main__":
    ipa_symbols = pd.read_csv("panphon_data/raw/ipa_symbols.csv", index_col="ipa")

    # if you only wish to add parts of the added symbols, just comment out the undesired methods.
    # diphthongs and triphthongs can also be added without distinctive lengths and nasalizations by
    # setting include_length and include_nasal to False.
    ipa_symbols = add_diphthongs(ipa_symbols)
    ipa_symbols = add_triphthongs(ipa_symbols)
    ipa_symbols = remove_bow(ipa_symbols)
    ipa_symbols = add_new_symbols(ipa_symbols)
    ipa_symbols = include_equivalent_symbols(ipa_symbols)

    ipa_symbols = ipa_symbols.fillna("0")
    ipa_symbols.to_csv("panphon_data/expanded/all_ipa_symbols.csv")
