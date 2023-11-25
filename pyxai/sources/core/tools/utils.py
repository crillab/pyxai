import platform
import random
import shap
import wordfreq
import importlib.util
import sys

from functools import reduce
from numpy import sum, mean
from operator import iconcat
from termcolor import colored
from time import time
from typing import Iterable

from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

from pyxai.sources.core.structure.type import PreferredReasonMethod


def check_PyQt6():
    ok, error = _check_PyQt6()
    if ok is False:
        print()
        print("The PyQt6 module is not installed or produces an import error.")
        print("Please use 'python3 -m pip install pyxai[gui]' to install this extras dependency.")
        raise error

def _check_PyQt6():
    try:
        import PyQt6
        from PyQt6.QtWidgets import QSplitter, QApplication, QAbstractItemView, QMessageBox, QFileDialog, QLabel, QSizePolicy, QScrollArea,  QMainWindow, QTableWidgetItem, QMenu, QGroupBox, QListWidget, QWidget, QVBoxLayout, QGridLayout, QTableWidget
        from PyQt6.QtGui import QAction, QPixmap, QColor
        from PyQt6.QtPrintSupport import QPrinter
    except ImportError as e:
        return False, e

    try:
        import PyQt6
        from PyQt6.QtWidgets import QSplitter, QApplication, QAbstractItemView, QMessageBox, QFileDialog, QLabel, QSizePolicy, QScrollArea,  QMainWindow, QTableWidgetItem, QMenu, QGroupBox, QListWidget, QWidget, QVBoxLayout, QGridLayout, QTableWidget
        from PyQt6.QtGui import QAction, QPixmap, QColor
        from PyQt6.QtPrintSupport import QPrinter
    except ModuleNotFoundError as e:
        return False, e
    return True, None
        
        
    

class Metric:

    @staticmethod
    def compute_metrics_regression(labels, predictions):
        return {
                "mean_squared_error": mean_squared_error(labels, predictions),
                "root_mean_squared_error": mean_squared_error(labels, predictions, squared=False),
                "mean_absolute_error": mean_absolute_error(labels, predictions)
            }

    @staticmethod
    def compute_metrics_binary_classification(labels, predictions):
        tp = Metric.compute_tp(labels, predictions)
        tn = Metric.compute_tn(labels, predictions)
        fp = Metric.compute_fp(labels, predictions)
        fn = Metric.compute_fn(labels, predictions)

        accuracy = ((tp+tn)/(tp+tn+fp+fn))*100
        precision = ((tp)/(tp+fp))*100 if tp+fp != 0 else 0
        recall = ((tp)/(tp+fn))*100 if tp+fn != 0 else 0
        f1score = (2*precision*recall)/(precision+recall) if precision+recall != 0 else 0
        specificity = ((tn)/(fp+tn))*100 if fp+tn != 0 else 0

        return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1score,
            "specificity": specificity,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "sklearn_confusion_matrix": confusion_matrix(labels, predictions).tolist()
            }
    
    @staticmethod
    def compute_metrics_multi_classification(labels, predictions, dict_labels):
        # TP, TN, FP and FN is not available the multi class case.
        # We therefore compute TP, TN, FP and FN for each class 
        true_positives = {}
        true_negatives = {}
        false_positives = {}
        false_negatives = {}
        accuracies = {}
        precisions = {}
        recalls = {}
        macro_averaging_accuracy = 0
        macro_averaging_precision = 0
        macro_averaging_recall = 0
        n_labels = len(dict_labels.keys())
        for key in dict_labels.keys():
            label = dict_labels[key]
            true_positives[key] = Metric.compute_tp(labels, predictions, label=label)
            true_negatives[key] = Metric.compute_tn(labels, predictions, label=label)
            false_positives[key] = Metric.compute_fp(labels, predictions, label=label)
            false_negatives[key] = Metric.compute_fn(labels, predictions, label=label)
            total = true_positives[key] + true_negatives[key] + false_positives[key] + false_negatives[key]
            if total != len(predictions):
                raise ValueError("The sum of TP, TN, FP and FN is not equal to the number of instances.")
            accuracies[key] = (true_positives[key] + true_negatives[key])/total
            if true_positives[key]+false_positives[key] == 0:
                precisions[key] = 1
            else:        
                precisions[key] = (true_positives[key])/(true_positives[key]+false_positives[key])
            if true_positives[key]+false_negatives[key] == 0:
                recalls[key] = 1
            else:
                recalls[key] = (true_positives[key])/(true_positives[key]+false_negatives[key])
            macro_averaging_accuracy += accuracies[key]
            macro_averaging_precision += precisions[key]
            macro_averaging_recall += recalls[key]
        
        macro_averaging_accuracy = (macro_averaging_accuracy/n_labels)*100
        macro_averaging_precision = (macro_averaging_precision/n_labels)*100
        macro_averaging_recall = (macro_averaging_recall/n_labels)*100

        micro_averaging_accuracy = 0
        micro_averaging_accuracy_numerator = 0
        micro_averaging_accuracy_denominator = 0
        micro_averaging_precision = 0
        micro_averaging_precision_numerator = 0
        micro_averaging_precision_denominator = 0
        micro_averaging_recall = 0
        micro_averaging_recall_numerator = 0
        micro_averaging_recall_denominator = 0

        for key in dict_labels.keys():
            label = dict_labels[key]
            micro_averaging_accuracy_numerator += true_positives[key]
            micro_averaging_accuracy_numerator += true_negatives[key]
            micro_averaging_accuracy_denominator += true_positives[key]
            micro_averaging_accuracy_denominator += true_negatives[key]
            micro_averaging_accuracy_denominator += false_positives[key]
            micro_averaging_accuracy_denominator += false_negatives[key]
            micro_averaging_precision_numerator += true_positives[key]
            micro_averaging_precision_denominator += true_positives[key]
            micro_averaging_precision_denominator += false_positives[key]
            micro_averaging_recall_numerator += true_positives[key]
            micro_averaging_recall_denominator += true_positives[key]
            micro_averaging_recall_denominator += false_negatives[key]
            

        micro_averaging_accuracy = (micro_averaging_accuracy_numerator/micro_averaging_accuracy_denominator)*100
        micro_averaging_precision = (micro_averaging_precision_numerator/micro_averaging_precision_denominator)*100
        micro_averaging_recall = (micro_averaging_recall_numerator/micro_averaging_recall_denominator)*100
        return {"micro_averaging_accuracy": micro_averaging_accuracy,
                "micro_averaging_precision": micro_averaging_precision,
                "micro_averaging_recall": micro_averaging_recall,
                "macro_averaging_accuracy": macro_averaging_accuracy,
                "macro_averaging_precision": macro_averaging_precision,
                "macro_averaging_recall": macro_averaging_recall,
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "accuracy": Metric.compute_accuracy_multiclass(labels, predictions),
                "sklearn_confusion_matrix": confusion_matrix(labels, predictions).tolist()
                }
    
    #A possible definition in the multi-class case is to take into account the well classified class.
    @staticmethod
    def compute_accuracy_multiclass(labels, predictions):
        well_classified = 0
        for i, _ in enumerate(predictions):
            if predictions[i] == labels[i]:
                well_classified += 1
        return (well_classified/len(predictions))*100
    

    # True Positive (TP): Cases where the prediction is positive, and the actual value is indeed positive.
    # Example: your doctor tells you that you're pregnant, and you are.
    # In the multi-class case, the `label` parameter have to be set by the good class.  
    # We consider this specific class as "positive" and all other classes as a single "negative" class.  
    @staticmethod
    def compute_tp(labels, predictions, label=None):
        tp = 0
        for i, _ in enumerate(predictions):
            if label is None:
                if predictions[i] == 1 and labels[i] == 1:
                    tp += 1
            else:
                if predictions[i] == label and labels[i] == label:
                    tp += 1
        return tp

    # True Negative (TN): Cases where the prediction is negative, and the actual value is actually negative. 
    # Example: the doctor tells you that you are not pregnant, and you are indeed not pregnant.
    # In the multi-class case, the `label` parameter have to be set by the good class.  
    # We consider this specific class as "positive" and all other classes as a single "negative" class.  
    @staticmethod
    def compute_tn(labels, predictions, label=None):
        tn = 0
        for i, _ in enumerate(predictions):
            if label is None:
                if predictions[i] == 0 and labels[i] == 0:
                    tn += 1
            else:
                if predictions[i] != label and labels[i] != label:
                    tn += 1
        return tn

    # False Positive (FP): Cases where the prediction is positive, but the actual value is negative. 
    # Example: the doctor tells that you're pregnant, but you're not.
    # In the multi-class case, the `label` parameter have to be set by the good class.  
    # We consider this specific class as "positive" and all other classes as a single "negative" class.  
    @staticmethod
    def compute_fp(labels, predictions, label=None):
        fp = 0
        for i, _ in enumerate(predictions):
            if label is None:
                if predictions[i] == 1 and labels[i] == 0:
                    fp += 1
            else:
                if predictions[i] == label and labels[i] != label:
                    fp += 1
        return fp

    # False Negative (FN): Cases where the prediction is negative, but the actual value is positive. 
    # Example: the doctor tells you that you're not pregnant, but you are.
    # In the multi-class case, the `label` parameter have to be set by the good class.  
    # We consider this specific class as "positive" and all other classes as a single "negative" class.  
    @staticmethod
    def compute_fn(labels, predictions, label=None):
        fn = 0
        for i, _ in enumerate(predictions):
            if label is None:
                if predictions[i] == 0 and labels[i] == 1:
                    fn += 1
            else:
                if predictions[i] != label and labels[i] == label:
                    fn += 1    
        return fn

    
class Stopwatch:
    def __init__(self):
        self.initial_time = time()

    def elapsed_time(self, *, reset=False):
        elapsed_time = time() - self.initial_time
        if reset:
            self.initial_time = time()
        return "{:.2f}".format(elapsed_time)

def switch_list(l, i1, i2):
    l[i1], l[i2] = l[i2], l[i1]
    return l

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
