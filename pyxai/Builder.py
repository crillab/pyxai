
from pyxai.sources.data_structures.decisionNode import DecisionNode
from pyxai.sources.data_structures.leafNode import LeafNode

from pyxai.sources.data_structures.decisionTree import DecisionTree as core_DecisionTree
from pyxai.sources.data_structures.randomForest import RandomForest as core_RandomForest
from pyxai.sources.data_structures.boostedTrees import BoostedTrees as core_BoostedTrees
from pyxai.sources.data_structures.boostedTreesRegression import BoostedTreesRegression as core_BoostedTreesRegression
from pyxai.sources.data_structures.linearModel import LinearModel, LinearModelSimple, LinearModelLogistic # TODO Remove in V2 Implement for V3

from pyxai.sources.data_structures.baseModel import BaseModel

from pyxai.sources.learners.learner_information import LearnerInformation



from pyxai.sources.type import OperatorCondition

GE = OperatorCondition.GE
GT = OperatorCondition.GT
LE = OperatorCondition.LE
LT = OperatorCondition.LT
EQ = OperatorCondition.EQ
NEQ = OperatorCondition.NEQ


class DecisionTree(core_DecisionTree):
    def __init__(self, n_features, root, n_classes=None, force_features_equal_to_binaries=False, learner_information=None, feature_names=None, class_names=None):
        if learner_information is None:
            learner_information = LearnerInformation(problem_type='classification', get_item_function = lambda x: x)
        elif learner_information.get_item_function is None:
            learner_information.get_item_function = lambda x: x
        super().__init__(n_features, root, n_classes, force_features_equal_to_binaries, learner_information, feature_names, class_names)


class RandomForest(core_RandomForest):
    def __init__(self, forest, n_features=None, n_classes=None, learner_information=None, feature_names=None, class_names=None):
        if learner_information is None:
            learner_information = LearnerInformation(problem_type='classification', get_item_function = lambda x: x)
        elif learner_information.get_item_function is None:
            learner_information.get_item_function = lambda x: x
        super().__init__(forest, n_features, n_classes, learner_information, feature_names, class_names)
        

class BoostedTrees(core_BoostedTrees):
    def __init__(self, forest, n_classes=None, learner_information=None, feature_names=None, class_names=None):
        if learner_information is None:
            learner_information = LearnerInformation(problem_type='classification', extras={"base_score": 0}, get_item_function = lambda x: x)
        elif learner_information.get_item_function is None:
            learner_information.get_item_function = lambda x: x
        super().__init__(forest, n_classes, learner_information, feature_names, class_names)


class BoostedTreesRegression(core_BoostedTreesRegression):
    def __init__(self, forest, learner_information=None, feature_names=None):
        if learner_information is None:
            learner_information = LearnerInformation(problem_type='regression', extras={"base_score": 0}, get_item_function = lambda x: x)
        elif learner_information.get_item_function is None:
            learner_information.get_item_function = lambda x: x
        super().__init__(forest, learner_information, feature_names)
