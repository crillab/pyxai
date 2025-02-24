import copy
import pickle

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from pyxai import Tools

from pyxai.sources.core.structure.decisionTree import DecisionTree, DecisionNode, LeafNode
from pyxai.sources.core.structure.randomForest import RandomForest
from pyxai.sources.learning.learner import Learner, NoneData
from pyxai.sources.core.structure.type import OperatorCondition, LearnerType, EvaluationOutput


class Scikitlearn(Learner):
    learners = {
        LearnerType.Classification: {
            EvaluationOutput.DT: DecisionTreeClassifier,
            EvaluationOutput.RF: RandomForestClassifier,
            EvaluationOutput.BT: None,
        },
        LearnerType.Regression: {
            EvaluationOutput.DT: None,
            EvaluationOutput.RF: None,
            EvaluationOutput.BT: None,
        },
    }
    
    def __init__(self, data=NoneData, *, learner_type=None, models_type=None):
        super().__init__(data, learner_type, models_type)
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

    """
    Fit a machine learning model and predict on test instances.

    This method trains a specified machine learning model using the provided training data and then
    uses the trained model to predict labels for the test data. It also computes performance metrics
    based on the predictions.

    Parameters:
    -----------
    learner_type : str
        The type of learner to use (e.g., 'classifier', 'regressor').
    output : str
        The specific output type for the learner (e.g., 'logistic_regression', 'random_forest').
    instances_training : array-like of shape (n_samples, n_features)
        The training input samples.
    instances_test : array-like of shape (n_samples, n_features)
        The test input samples.
    labels_training : array-like of shape (n_samples,)
        The target values (class labels in classification, continuous values in regression) for the training samples.
    labels_test : array-like of shape (n_samples,)
        The true target values for the test samples.
    learner_options : dict
        A dictionary of options to pass to the learner. If 'seed' is provided, it will be converted to 'random_state'.

    Returns:
    --------
    tuple
        A tuple containing:
        - learner : object
            The trained learner object.
        - metrics : dict
            A dictionary of performance metrics computed on the test data.
        - extras : dict
            Additional information including the type of learner, learner options, and a base score.

    Raises:
    -------
    NotImplementedError
        If the specified learner type and output combination is not implemented for Scikit-Learn.

    Notes:
    ------
    The method internally converts 'seed' in learner_options to 'random_state' if present, as Scikit-Learn uses 'random_state'.
    """   
    def fit_and_predict(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        learner = Scikitlearn.learners[self.learner_type][self.models_type]
        if learner is None: 
            raise NotImplementedError(str(self.models_type) + " and " + str(self.learner_type) + "is not implemented for ScikitLearn.")
        
        if "seed" in learner_options.keys():
            learner_options["random_state"] = learner_options["seed"]
            learner_options.pop("seed")
        
        self.display_parameters(learner_options)
        learner = learner(**learner_options)
        learner.fit(instances_training, labels_training)
       
        result = learner.predict(instances_test)
        metrics = self.compute_metrics(labels_test, result)

        extras = {
            "learner": str(type(learner)),
            "learner_options": learner_options,
            "base_score": 0,
        }
        return (copy.deepcopy(learner), metrics, extras)
    
    """
    Convert the model to a specific learner type and output format.

    This method converts the current model to a specified learner type and output format.
    It raises a `NotImplementedError` if the specified combination of learner type and output
    is not supported by ScikitLearn.

    Parameters:
    -----------
    learner_type : str
        The type of learner to convert the model to. This should be a key in the `learners` dictionary.
    output : EvaluationOutput
        The output format to convert the model to. This should be a member of the `EvaluationOutput` enum.
    learner_information : dict, optional
        Additional information about the learner. If provided, this information will be stored in the
        `learner_information` attribute of the instance.

    Returns:
    --------
    model : object
        The converted model in the specified learner type and output format.

    Raises:
    -------
    NotImplementedError
        If the specified combination of `learner_type` and `output` is not implemented for ScikitLearn.
    """
    def convert_model(self, learner_information=None):
        learner = Scikitlearn.learners[self.learner_type][self.models_type]
        if learner is None: 
            raise NotImplementedError(str(self.models_type) + " and " + str(self.learner_type) + "is not implemented for ScikitLearn.")
              
        if learner_information is not None:
            self.learner_information = learner_information 
        
        if self.models_type == EvaluationOutput.DT:
            return self.to_decision_tree()
        elif self.models_type == EvaluationOutput.RF:
            return self.to_random_forests()
        else:
            raise NotImplementedError(str(self.models_type) + " and " + str(self.learner_type) + "is not implemented for ScikitLearn.")
        

    def to_decision_tree(self):
        decision_trees = []
        for id_solver_results, _ in enumerate(self.learner_information):
            sk_tree = self.learner_information[id_solver_results].raw_model
            sk_raw_tree = sk_tree.tree_
            decision_trees.append(self.sk_tree_to_tree(sk_tree, sk_raw_tree, id_solver_results))
        return decision_trees

    def to_random_forests(self):
        random_forests = []
        for id_solver_results, _ in enumerate(self.learner_information):
            random_forest = self.learner_information[id_solver_results].raw_model
            decision_trees = []
            for sk_tree in random_forest:
                sk_raw_tree = sk_tree.tree_
                decision_trees.append(self.sk_tree_to_tree(sk_tree, sk_raw_tree, id_solver_results))
            random_forests.append(
                RandomForest(decision_trees, n_classes=len(sk_tree.classes_), learner_information=self.learner_information[id_solver_results]))
        return random_forests

    """
    Convert a specific Scikitlearn's decision tree into a program-specific object called 'DecisionTree'.
    """
    def sk_tree_to_tree(self, sk_tree, sk_raw_tree, id_solver_results=0):
        """
        Warning: we use here numpy.argmax(sk_raw_tree.value[0][0]) to get the predition of a tree. 
        But sklearn do not that of this way.
        - For us, a leaf of a tree represents a class
        - Not for sklearn where a leaf is a list of probability of classes (example: [0.3 0.4 0.6] for 3 classes)
        This is can change the prediction between our model and the sklearn model. 
        - For us, we take the argmax of the number of trees that predict a class.
        - sklearn does a average of lists of probabilities. 
        However, the predictions are 99% identical.  
        """
        nodes = {i: DecisionNode(int(feature + 1), threshold=sk_raw_tree.threshold[i], operator=OperatorCondition.GT, left=None, right=None)
                 for i, feature in enumerate(sk_raw_tree.feature) if feature >= 0}
        for i in range(len(sk_raw_tree.feature)):
            if i in nodes:
                # Set left and right of each node
                id_left = sk_raw_tree.children_left[i]
                id_right = sk_raw_tree.children_right[i]

                nodes[i].left = nodes[id_left] if id_left in nodes else LeafNode(numpy.argmax(sk_raw_tree.value[id_left][0]))
                nodes[i].right = nodes[id_right] if id_right in nodes else LeafNode(numpy.argmax(sk_raw_tree.value[id_right][0]))
        
        root = nodes[0] if 0 in nodes else LeafNode(numpy.argmax(sk_raw_tree.value[0][0]))
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
