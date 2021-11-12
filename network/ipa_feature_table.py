class IPAFeatureTable:
    def __init__(self, fp="data/all_ipa_symbols.csv"):
        self.feature_table = dict()
        try:
            with open(fp, "r") as f:
                for line in f:
                    fields = line.split(",")
                    ipa = fields[0]
                    if ipa == "":
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
        except:
            FileNotFoundError()

    def __call__(self, symbol):
        return self.feature_table[symbol]

    def get_ipa_list(self):
        return self.feature_table.keys()

    def get_pair_encoding(self, symbol1, symbol2):
        if symbol1 == "-":
            return self.feature_table[symbol2]
        if symbol2 == "-":
            return self.feature_table[symbol1]

        symbol1_features = self.feature_table[symbol1]
        symbol2_features = self.feature_table[symbol2]
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

    def get_equivalence_classes(self, fp=None, mask=None):
        eq_classes = dict()
        for sound in self.feature_table:
            features = self.feature_table[sound].copy()
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
