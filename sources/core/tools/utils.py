import platform
import random
import shap
import wordfreq
from functools import reduce
from numpy import sum, mean
from operator import iconcat
from termcolor import colored
from time import time
from typing import Iterable

from pyxai.sources.core.structure.type import PreferredReasonMethod


class Stopwatch:
    def __init__(self):
        self.initial_time = time()


    def elapsed_time(self, *, reset=False):
        elapsed_time = time() - self.initial_time
        if reset:
            self.initial_time = time()
        return "{:.2f}".format(elapsed_time)


def flatten(lit):
    return reduce(iconcat, lit, [])


def count_dimensions(lit):
    n = 0
    tmp = lit
    while True:
        if isinstance(tmp, Iterable):
            n = n + 1
            tmp = tmp[0]
        else:
            break
    return n


_verbose = 1


def set_verbose(v):
    global _verbose
    v = int(v)
    _verbose = v


def verbose(*message):
    global _verbose
    if _verbose > 0:
        print(*message)


def get_os():
    return platform.system().lower()


def shuffle(_list):
    random.shuffle(_list)
    return _list


def add_lists_by_index(list1, list2):
    """
    Adding two lists results in a new list where each element is the sum of the elements in the corresponding positions of the two lists.
    """
    return [x + y for (x, y) in zip(list1, list2)]


def compute_accuracy(prediction, right_prediction):
    return (sum(prediction == right_prediction) / len(right_prediction)) * 100


def display_observation(observation, size=28):
    for i, element in enumerate(observation):
        print(colored('X', 'blue') if element == 0 else colored('X', 'red'), end='')
        if (i + 1) % size == 0 and i != 0:
            print()


def compute_weight(method, instance, weights, learner_information, *, features_partition = None):
    if method is None:
        raise ValueError(
            "The method parameter is not correct (possible choice: Minimal, Weights, Shapely, FeatureImportance, WordFrequency, WordFrequencyLayers")

    if method == PreferredReasonMethod.Minimal:
        return [1 for _ in range(len(instance))]

    if method == PreferredReasonMethod.Weights:
        # weights: the weights chosen by the user, which correspond to the user's preference
        if isinstance(weights, list):
            return [-weights[i] + 1 + max(weights) for i in range(len(weights))]
        if isinstance(weights, dict):
            feature_names = learner_information.feature_names
            new_weights = [0 for _ in range(len(feature_names))]
            for i in weights.keys():
                new_weights[abs(i)] = weights[i]
            return [-new_weights[i] + 1 + max(new_weights) for i in range(len(new_weights))]
        raise ValueError("The 'weights' parameter is not correct (must be a list a feature weights or a dict of features weights).")

    if method == PreferredReasonMethod.Shapley:
        # Shapely values for the model trained on data.
        raw_model = learner_information.raw_model

        shapely_explainer = shap.TreeExplainer(raw_model, model_output='raw')
        shapely_values = shapely_explainer.shap_values(instance, check_additivity=False)
        shapely_value = mean(shapely_values, axis=0) if len(shapely_values) > 2 else shapely_values[0]
        # Decreasing monotonous affine transformation for shapely values, w = alpha * x + b , where x = shapely_value
        # alpha = - min(a, 1) where a is the minimum value greater than zero of abs (shapely_value)
        alpha = min([i for i in abs(shapely_value) if i != 0])
        shapely_value = -shapely_value / min(alpha, 1)
        return [round(shapely_value[i] - min(shapely_value) + 1) for i in range(len(shapely_value))]

    if method == PreferredReasonMethod.FeatureImportance:
        # Feature importance from sklearn
        # Decreasing monotonous affine transformation
        raw_model = learner_information.raw_model
        feature_importances = raw_model.feature_importances_
        alpha = min([i for i in feature_importances if i != 0])  # the minimum value greater than zero
        feature_importances = -10 * feature_importances / alpha  # alpha = -10/a, b = 1
        return [round(feature_importances[i] - min(feature_importances) + 1) for i in range(len(feature_importances))]

    if method == PreferredReasonMethod.WordFrequency:
        feature_names = learner_information.feature_names
        weights_feature = [int(wordfreq.zipf_frequency(name, 'en') * 100 * (wordfreq.zipf_frequency(name, 'en') > 0)
                               + (wordfreq.zipf_frequency(name, 'en') == 0)) for name in feature_names]
        return [-weights_feature[i] + 1 + max(weights_feature) for i in range(len(weights_feature))]

    if method == PreferredReasonMethod.WordFrequencyLayers:
        feature_names = learner_information.feature_names
        weights = [0 for _ in range(len(feature_names))]
        a = [0 for _ in range(3)]  # number of layers - 1
        for i in range(len(feature_names)):
            if 6 < wordfreq.zipf_frequency(feature_names[i], 'en'):
                a[0] += 1
                weights[i] = 1
            if 4 < wordfreq.zipf_frequency(feature_names[i], 'en') <= 6:
                a[1] += 1
                weights[i] = 10 + a[0]  # for non-compensation
            if 2 < wordfreq.zipf_frequency(feature_names[i], 'en') <= 4:
                a[2] += 1
                weights[i] = 20 + a[0] + 10 * a[1]  # for non-compensation
            if 0 <= wordfreq.zipf_frequency(feature_names[i], 'en') <= 2:
                weights[i] = 100 + a[0] + 10 * a[1] + 20 * a[2]  # for non-compensation
        return weights

    if method == PreferredReasonMethod.InclusionPreferred:
        tmp = tmp2 = 1
        _dict = {}
        i = 0
        weights = [0] * len(learner_information.feature_names)
        for f in learner_information.feature_names:
            _dict[f] = i
            i = i + 1
        if features_partition is None or len(_dict) != len(learner_information.feature_names):
            raise ValueError("The partiion_features field must contain all features (as a partition)")
        for _set in features_partition:
            for f in _set:
                tmp2 += tmp
                weights[_dict[f]] = tmp
            tmp = tmp2
        return weights
    raise ValueError("The method parameter is not correct (Minimal, Weights, Shapely, FeatureImportance, WordFrequency, WordFrequencyLayers).")
