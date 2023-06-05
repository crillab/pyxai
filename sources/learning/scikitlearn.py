import copy
import pickle

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from pyxai import Tools

from pyxai.sources.core.structure.decisionTree import DecisionTree, DecisionNode, LeafNode
from pyxai.sources.core.structure.randomForest import RandomForest
from pyxai.sources.core.tools.utils import compute_accuracy
from pyxai.sources.learning.learner import Learner, NoneData
from pyxai.sources.core.structure.type import OperatorCondition, LearnerType, EvaluationOutput

class Scikitlearn(Learner):
    def __init__(self, data=NoneData, *, learner_type=None):
        super().__init__(data, learner_type)
        self.has_to_display_parameters = True

    def display_parameters(self, learner_options):
        if self.has_to_display_parameters is True:
            Tools.verbose("learner_options:", learner_options)
            self.has_to_display_parameters = False
    
    @staticmethod
    def get_learner_types():
        return {type(DecisionTreeClassifier()): (LearnerType.Classification, EvaluationOutput.DT),
                type(RandomForestClassifier()): (LearnerType.Classification, EvaluationOutput.RF)}

    @staticmethod
    def get_learner_name():
        return str(Scikitlearn.__name__)

    def fit_and_predict_DT_CLS(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        if "seed" in learner_options.keys():
            learner_options["random_state"] = learner_options["seed"]
            learner_options.pop("seed")

        self.display_parameters(learner_options)
        learner = DecisionTreeClassifier(**learner_options)
        learner.fit(instances_training, labels_training)

        result = learner.predict(instances_test)
        metrics = self.compute_metrics(labels_test, result)

        extras = {
            "learner": str(type(learner)),
            "learner_options": learner_options,
            "base_score": 0,
        }
        return (copy.deepcopy(learner), metrics, extras)


    def fit_and_predict_RF_CLS(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        if "seed" in learner_options.keys():
            learner_options["random_state"] = learner_options["seed"]
            learner_options.pop("seed")

        self.display_parameters(learner_options)
        learner = RandomForestClassifier(**learner_options)
        learner.fit(instances_training, labels_training)
        
        result = learner.predict(instances_test)
        metrics = self.compute_metrics(labels_test, result)
        extras = {
            "learner": str(type(learner)),
            "learner_options": learner_options,
            "base_score": 0,
        }
        return (copy.deepcopy(learner), metrics, extras)
    

    def fit_and_predict_BT_CLS(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Boosted Trees with classification is not implemented for ScikitLearn.")

    def fit_and_predict_DT_REG(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Boosted Trees with regression is not implemented for ScikitLearn.")
    
    def fit_and_predict_RF_REG(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Boosted Trees with regression is not implemented for ScikitLearn.")
    
    def fit_and_predict_BT_REG(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Boosted Trees with regression is not implemented for ScikitLearn.")

    """
    Convert the Scikitlearn's decision trees into the program-specific objects called 'DecisionTree'.
    """


    def to_DT_CLS(self, learner_information=None):
        decision_trees = []
        for id_solver_results, _ in enumerate(self.learner_information):
            sk_tree = self.learner_information[id_solver_results].raw_model
            sk_raw_tree = sk_tree.tree_
            decision_trees.append(self.classifier_to_DT(sk_tree, sk_raw_tree, id_solver_results))
        return decision_trees

    def to_RF_CLS(self, learner_information=None):
        random_forests = []
        for id_solver_results, _ in enumerate(self.learner_information):
            random_forest = self.learner_information[id_solver_results].raw_model
            decision_trees = []
            for sk_tree in random_forest:
                sk_raw_tree = sk_tree.tree_
                decision_trees.append(self.classifier_to_DT(sk_tree, sk_raw_tree, id_solver_results))
            random_forests.append(
                RandomForest(decision_trees, n_classes=len(sk_tree.classes_), learner_information=self.learner_information[id_solver_results]))
        return random_forests


    def to_BT_CLS(self, learner_information=None):
        raise NotImplementedError("Boosted Trees with classification is not implemented for ScikitLearn.")

    def to_DT_REG(self, learner_information=None):
        raise NotImplementedError("Boosted Trees with regression is not implemented for ScikitLearn.")

    def to_RF_REG(self, learner_information=None):
        raise NotImplementedError("Boosted Trees with regression is not implemented for ScikitLearn.")

    def to_BT_REG(self, learner_information=None):
        raise NotImplementedError("Boosted Trees with regression is not implemented for ScikitLearn.")


   

    """
    Convert a specific Scikitlearn's decision tree into a program-specific object called 'DecisionTree'.
    """

    def classifier_to_DT(self, sk_tree, sk_raw_tree, id_solver_results=0):
        nodes = {i: DecisionNode(int(feature + 1), threshold=sk_raw_tree.threshold[i], operator=OperatorCondition.GT, left=None, right=None)
                 for i, feature in enumerate(sk_raw_tree.feature) if feature >= 0}
        for i in range(len(sk_raw_tree.feature)):
            if i in nodes:
                # Set left and right of each node
                id_left = sk_raw_tree.children_left[i]
                id_right = sk_raw_tree.children_right[i]

                nodes[i].left = nodes[id_left] if id_left in nodes else LeafNode(numpy.argmax(sk_raw_tree.value[id_left][0]))
                nodes[i].right = nodes[id_right] if id_right in nodes else LeafNode(numpy.argmax(sk_raw_tree.value[id_right][0]))
                
        root = nodes[0] if 0 in nodes else DecisionNode(1, 0, sk_raw_tree.value[0][0])
        return DecisionTree(sk_tree.n_features_in_, root, sk_tree.classes_, id_solver_results=id_solver_results,
                            learner_information=self.learner_information[id_solver_results])

    def save_model(self, learner_information, filename):
        file = open(filename + ".model", 'wb')
        pickle.dump(learner_information.raw_model, file)
        file.close()

    def load_model(self, model_file, learner_options):
        learner = None
        with open(model_file, 'rb') as file:
            learner = pickle.load(file)
        return learner
