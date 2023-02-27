import numpy as np
from models.storage import load_change_model
from features.ipa_feature_table import *
from itertools import product
from scipy.special import softmax

ipa_feature_table = IPAFeatureTable()


def calculate_logits(source_sounds, target_sounds, model):
    source_sounds = set(source_sounds)
    logits_by_source = {}
    logits_by_target = {}

    for source, target in product(source_sounds, target_sounds):
        x = np.array([ipa_feature_table(source) + ipa_feature_table(target)])
        logit = float(model(x))

        source_logits = logits_by_source[source] if source in logits_by_source else {}
        target_logits = logits_by_target[target] if target in logits_by_target else {}

        source_logits[target] = logit
        target_logits[source] = logit

        logits_by_source[source] = source_logits
        logits_by_target[target] = target_logits

    return logits_by_source, logits_by_target


def softmax_distribution(logits):
    prob_distributions = {}

    for s1 in logits:
        logits_for_sound = logits[s1]
        sounds, values = [], []
        for s2, v in logits_for_sound.items():
            sounds.append(s2)
            values.append(v)

        dist = softmax(values)
        dist_dict = {}
        for s2, prob in zip(sounds, dist):
            dist_dict[s2] = prob

        prob_distributions[s1] = dist_dict

    return prob_distributions


def transpose(probs):
    transposed_probs = {}
    outer_keys = list(probs.keys())
    inner_keys = list(probs[outer_keys[0]].keys())

    for inner in inner_keys:
        inner_dict = {}
        for outer in outer_keys:
            inner_dict[outer] = probs[outer][inner]
        transposed_probs[inner] = inner_dict

    return transposed_probs


def normalize_by_source(source_sounds, target_sounds, probs):
    source_sounds = set(source_sounds)
    source_probs = {}
    source_prob_mass = {}

    for source in source_sounds:
        prob_mass = 1
        for target in target_sounds:
            unnormalized_prob = probs[source][target]
            prob_mass *= unnormalized_prob
        source_prob_mass[source] = prob_mass

    for source, prob_mass in source_prob_mass.items():
        normalized_prob = prob_mass / sum(source_prob_mass.values())
        source_probs[source] = normalized_prob

    return source_probs


if __name__ == "__main__":
    model = load_change_model("resources/models/change/trained_models/change-model-0.hdf5")

    sources = ["p", "f", "x", "h", "ⁿpʰ", "ⁿpʰ", "m̥", "bʷ", "v", "pʰ", "b", "m"]
    # sources = ["ⁿpʰ", "ⁿpʰ", "m̥", "bʷ", "v", "pʰ", "b", "m", "p", "f"]
    # sources.extend(8 * ["ⁿpʰ"])
    targets = ["p", "f", "x", "h"]

    logits_by_source, logits_by_target = calculate_logits(sources, targets, model)
    probs_by_target = softmax_distribution(logits_by_target)
    probs_by_source = transpose(probs_by_target)
    # probs_by_source = softmax_distribution(logits_by_source)
    normalized_probs = normalize_by_source(sources, targets, probs_by_source)
    for s, prob in normalized_probs.items():
        print(f"{s}:\t{prob:.4f}")
