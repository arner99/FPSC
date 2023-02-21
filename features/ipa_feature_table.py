class IPAFeatureTable:
    def __init__(self, fp="data/features/base_ipa_symbols.csv",
                 diac_fp="data/features/diacritic_rules.csv",
                 vow_fp="data/features/vowel_dimensions.csv"):
        self.feature_table = dict()
        self.modifier_table = dict()
        self.vowel_dimensions = dict()
        try:
            with open(fp, "r") as f:
                for line in f:
                    fields = line.split(",")
                    ipa = fields[0]
                    if ipa == "" or ipa == "ipa":
                        self.features = fields[1:]
                        continue
                    features = fields[1:]
                    num_features = []
                    for feature in features:
                        feature = feature.replace("\n", "")
                        if feature == "+":
                            num_features.append(1)
                        elif feature == "0":
                            num_features.append(0)
                        elif feature == "-":
                            num_features.append(-1)
                        else:
                            raise ValueError()
                    self.feature_table[ipa] = num_features
            with open(diac_fp, "r") as f:
                for line in f:
                    fields = line.split()
                    diac = fields[0]
                    if len(fields) > 1:
                        modifications = fields[1].split(",")
                    else:
                        modifications = []
                    self.modifier_table[diac] = modifications
            with open(vow_fp, "r") as f:
                for line in f:
                    fields = line.split()
                    vowel = fields[0]
                    dimensions = [float(x) for x in fields[1].split(",")]
                    self.vowel_dimensions[vowel] = dimensions
        except:
            FileNotFoundError()

    def __call__(self, symbol):
        return self.get(symbol)

    def __contains__(self, item):
        return self.get(item) is not None

    def get(self, symbol):
        if self.is_polyphthong(symbol):
            reordered_polyphthong = self.handle_polyphthong(symbol)
            symbol = reordered_polyphthong
        return self.handle_diacritic(symbol)

    def is_polyphthong(self, symbol):
        vowel_count = 0
        for c in symbol:
            if c in self.vowel_dimensions:
                vowel_count += 1

        return vowel_count == 2 or vowel_count == 3

    def handle_polyphthong(self, symbol):
        polyphthong = ""
        other_symbols = ""
        for c in symbol:
            if c == '͡' or c == '̯' or c in other_symbols:
                continue

            if c in self.vowel_dimensions:
                polyphthong += c
            else:
                other_symbols += c

        if len(polyphthong) == 2:
            self.encode_diphthong(polyphthong)
        if len(polyphthong) == 3:
            self.encode_triphthong(polyphthong)

        return polyphthong + other_symbols

    def encode_diphthong(self, diphthong):
        if len(diphthong) != 2:
            return None

        first_vowel = diphthong[0]
        second_vowel = diphthong[1]

        if (first_vowel not in self.vowel_dimensions or second_vowel not in self.vowel_dimensions
                or first_vowel not in self.feature_table or second_vowel not in self.feature_table):
            return None

        first_vowel_dimensions = self.vowel_dimensions[first_vowel]
        second_vowel_dimensions = self.vowel_dimensions[second_vowel]
        feature_vector = self.feature_table[first_vowel].copy()

        # BACKSHIFT / FRONTSHIFT
        if second_vowel_dimensions[0] > first_vowel_dimensions[0]:
            feature_vector[24] = 1
            feature_vector[25] = -1
        elif second_vowel_dimensions[0] < first_vowel_dimensions[0]:
            feature_vector[24] = -1
            feature_vector[25] = 1
        else:
            feature_vector[24] = -1
            feature_vector[25] = -1

        # OPENING / CLOSING
        if abs(first_vowel_dimensions[1] - second_vowel_dimensions[1]) > 0.8:
            if second_vowel_dimensions[1] > first_vowel_dimensions[1]:  # opening
                feature_vector[26] = 1
                feature_vector[28] = -1
            else:  # closing
                feature_vector[26] = -1
                feature_vector[28] = 1
        else:
            feature_vector[26] = -1
            feature_vector[28] = -1

        # CENTERING
        if ((second_vowel_dimensions[0] == 2 and 2 <= second_vowel_dimensions[1] <= 3) or
                second_vowel_dimensions[1] == 1.5 and first_vowel_dimensions[1] != 1.5 or
                second_vowel_dimensions[1] == 3.5 and first_vowel_dimensions[1] != 3.5 or
                second_vowel_dimensions[1] == 2 and first_vowel_dimensions[1] < 2 or
                second_vowel_dimensions[1] == 3 and first_vowel_dimensions[1] > 3):
            feature_vector[27] = 1
        else:
            feature_vector[27] = -1

        # LONGDISTANCE
        if 2 <= first_vowel_dimensions[1] <= 3:
            if abs(first_vowel_dimensions[1] - second_vowel_dimensions[1]) == 2:
                feature_vector[29] = 1
            else:
                feature_vector[29] = -1
        else:
            if abs(first_vowel_dimensions[1] - second_vowel_dimensions[1]) > 2:
                feature_vector[29] = 1
            else:
                feature_vector[29] = -1

        # SECONDROUNDED
        if second_vowel_dimensions[2] == 1:
            feature_vector[30] = 1
        else:
            feature_vector[30] = -1

        self.feature_table[diphthong] = feature_vector

        return feature_vector

    def encode_triphthong(self, triphthong):
        if len(triphthong) != 3:
            return None

        leading_diphthong = triphthong[:2]
        closing_diphthong = triphthong[1:]

        feature_vector = self.encode_diphthong(leading_diphthong).copy()
        closing_diphthong_features = self.encode_diphthong(closing_diphthong)

        if not feature_vector or not closing_diphthong_features:
            return None

        for i in range(24, 30):
            if closing_diphthong_features[i] == 1:
                feature_vector[i] = 1

        feature_vector[30] = closing_diphthong_features[30]

        if closing_diphthong_features[18] == 1:
            feature_vector[18] = 1

        self.feature_table[triphthong] = feature_vector

        return feature_vector

    def handle_diacritic(self, symbol):
        if symbol in self.feature_table:
            return self.feature_table[symbol]
        else:
            if len(symbol) <= 1:
                return None

            # try to parse last character as modifying diacritic,
            # if last character is no known diacritic try with first character
            if symbol[-1] in self.modifier_table:
                modifier = symbol[-1]
                remaining_symbol = symbol[:-1]
            elif symbol[0] in self.modifier_table:
                modifier = symbol[0]
                remaining_symbol = symbol[1:]
            else:
                return None

            base_feature_vector = self.handle_diacritic(remaining_symbol)

            if not base_feature_vector:
                return None

            feature_vector = base_feature_vector.copy()

            for modification in self.modifier_table[modifier]:
                if modification == "":
                    continue
                modified_feature = modification[1:]
                feature_idx = self.features.index(modified_feature)
                if modification[0] == "+":
                    feature_vector[feature_idx] = 1
                elif modification[0] == "-":
                    feature_vector[feature_idx] = -1
                else:
                    return None

            self.feature_table[symbol] = feature_vector

            return feature_vector

    def get_cached_symbols(self):
        return self.feature_table.keys()

    """
    pair encoding as expected by the symmetrical sound correspondence model.
    for each feature, one of the following values is assigned:
    1  - if the feature is applicable and matches for both sounds (+ or - for both)
    0  - if the feature does not apply for both sounds
    -1 - in all other cases, i.e. when the feature is applicable for at least one sound and does not match
    """
    def get_pair_encoding(self, symbol1, symbol2):
        if symbol1 == "-":
            return self.get(symbol2)
        if symbol2 == "-":
            return self.get(symbol1)

        symbol1_features = self.get(symbol1)
        symbol2_features = self.get(symbol2)

        if not (symbol1_features and symbol2_features):
            raise KeyError

        feature_classes = []
        for i in range(len(symbol1_features)):
            feat1 = symbol1_features[i]
            feat2 = symbol2_features[i]
            if (feat1 == 1 and feat2 == 1) or (feat1 == -1 and feat2 == -1):
                feature_classes.append(1)
            elif feat1 == 0 and feat2 == 0:
                feature_classes.append(0)
            else:
                feature_classes.append(-1)

        return feature_classes

    def get_equivalence_classes(self, alphabet, fp=None, mask=None):
        if not alphabet:
            alphabet = self.feature_table.keys()

        eq_classes = dict()
        for sound in alphabet:
            features = self.get(sound).copy()
            if mask is not None:
                try:
                    features.pop(mask)
                except:
                    print(len(features), mask)
            features = " ".join(map(str, features))
            if features in eq_classes:
                eq_classes[features].append(sound)
            else:
                eq_classes[features] = [sound]
        eq_classes_list = list(eq_classes.values())
        if fp is not None:
            with open(fp, "w") as f:
                for eq_class in eq_classes_list:
                    f.write(" ".join(eq_class))
                    f.write("\n")
