import pickle
from typing import Callable
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import linear_model
from copy import deepcopy
import inspect

from pyxai import Tools
from pyxai.sources.data_structures.linearModel import LinearModel, LinearModelSimple, LinearModelLogistic
from pyxai.sources.data_structures.decisionTree import DecisionTree, DecisionNode, LeafNode
from pyxai.sources.data_structures.randomForest import RandomForest
from pyxai.sources.data_structures.baseModel import BaseModel
from pyxai.sources.type import OperatorCondition, ProblemType, ModelType, InstancesType, LabelsType
from pyxai.sources.learners.learner import Learner, NoneData


class Scikitlearn(Learner):
    r""" This class is used to create a model using the [Scikit-learn](https://scikit-learn.org/stable/) library. 
    All the methods used in this class (```evaluate```, ```get_instances```, ...) are documented in the parent class ```Learner```.

    """
    models = {
        ProblemType.Classification: {
            ModelType.DecisionTree: DecisionTreeClassifier,
            ModelType.RandomForest: RandomForestClassifier,
            ModelType.BoostedTree: None,
            ModelType.LinearSimple: None,
            ModelType.LinearLogistic: linear_model.LogisticRegression,
            ModelType.LinearRidge: linear_model.RidgeClassifier,
            ModelType.LinearLasso: None,
        },
        ProblemType.Regression: {
            ModelType.DecisionTree: None,
            ModelType.RandomForest: None,
            ModelType.BoostedTree: None,
            ModelType.LinearSimple: linear_model.LinearRegression,
            ModelType.LinearLogistic: None,
            ModelType.LinearRidge: linear_model.Ridge,
            ModelType.LinearLasso: None,
        },
    }
    
    # /!\ Docstring of this method come from the parent class Learner.  
    def __init__(self, data: str|pandas.DataFrame|NoneData = NoneData, *,
                 problem_type: str|ProblemType|None = None,
                 model_type: str|ModelType|None = None,
                 instances_type: str|InstancesType|None = None,
                 labels_type: str|LabelsType|None = None,
                 get_item_function: Callable|None = None,
                 instances_directory: str|None = None,
                 labels_directory: str|None = None):
        r"""Initialise the learner with the dataset and its characteristics. 
            The dataset can be given as a path to a csv, json or excel file or as a pandas DataFrame. 
            In the case of a csv or excel file, the last column is considered as the label column and all other columns are considered as features. 
            In the case of a json file, the dataset should be in a specific format (see documentation). 

            Parameters
            ----------
            data: ```str``` | ```pandas.DataFrame``` | ```NoneData```
                The dataset to use, either as a path to a csv, json or excel file or as a pandas DataFrame.

            problem_type: ```str``` | ```ProblemType```
                The type of problem (classification, regression, ...)
                Possible values are defined in the ```ProblemType``` enum.
            
            model_type: ```str``` | ```ModelType``` (optional, default=```None```)
                The type of model (linear, tree-based, neural network, ...)
                Can be ```None``` and put in the evaluation method. 
                Possible values are defined in the ```ModelType``` enum.

            instances_type: ```str``` | ```InstancesType``` (optional, default=```None```)
                The type of instances (image, tabular, text, temporal, ...)
                Possible values are defined in the ```InstancesType``` enum.
                When set to None, if data is a csv and get_item_function is None then it is set to "tabular"
                    
            labels_type: ```str``` | ```LabelsType``` (optional, default=```None```)
                The type of labels (class, text, mask, contours, ...)
                Possible values are defined in the ```LabelsType``` enum.
                When set to None, if problem_type is "classification" then it is set to "classes", if problem_type is "regression" then it is set to "continuous-values"

            get_item_function: ```Callable``` (optional, default=```None```)
                A function to get an instance from the dataset. This function is used to get an instance in the right format for the model and the explainer.
                If the dataset is a pandas DataFrame and the instances are tabular, this function is not necessary and can be set to None. 
                In other cases, this function should be defined by the user. It should take as input a row of the dataframe and return the corresponding instance in the right format for the model and the explainer.

            instances_directory: ```str``` (optional, default=```None```)
                The directory where the instances are stored (only for a JSON dataset). 
                This parameter is used to extend the path of instances in the dataframe when the instances are of type image and the dataset is given as a json file.
            
            labels_directory: ```str``` (optional, default=```None```)
                The directory where the labels are stored (only for a JSON dataset). 
                This parameter is used to extend the paths of labels in the dataframe when the labels are of type masks and the dataset is given as a json file.
                Warning: NOT YET IMPLEMENTED

            Returns
            -------
            ```Scikitlearn```:
               A Learner object.

            Examples
            --------
            >>> from pyxai import Learning, Tools
            >>> learner = Learning.Scikitlearn(Tools.Options.dataset, problem_type=Learning.CLASSIFICATION)
            >>> model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF)
            >>> instance, prediction = learner.get_instances(n=1)
            
            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/learning/generating/)
        """
        super().__init__(data, problem_type=problem_type, instances_type=instances_type, labels_type=labels_type,
                         model_type=model_type, instances_directory=instances_directory, labels_directory=labels_directory,
                         get_item_function=get_item_function)
        self.has_to_display_parameters = True

    def display_parameters(self, model_parameters):
        if self.has_to_display_parameters is True:
            Tools.verbose("model_parameters:", model_parameters)
            self.has_to_display_parameters = False
    
    @staticmethod
    def get_problem_types():
        return {type(DecisionTreeClassifier()): (ProblemType.Classification, ModelType.DecisionTree),
                type(RandomForestClassifier()): (ProblemType.Classification, ModelType.RandomForest)}

    @staticmethod
    def get_learner_name():
        return str(Scikitlearn.__name__)

    """
    Fit a machine learning model and predict on test instances.

    This method trains a specified machine learning model using the provided training data and then
    uses the trained model to predict labels for the testing data. It also computes performance metrics
    based on the predictions.

    Parameters:
    -----------
    problem_type : str|ProblemType
        The type of model to use (e.g., 'classifier', 'regressor').
    output : str
        The specific output type for the model (e.g., 'logistic_regression', 'random_forest').
    instances_train : array-like of shape (n_samples, n_features)
        The train input samples.
    instances_test : array-like of shape (n_samples, n_features)
        The test input samples.
    labels_train : array-like of shape (n_samples,)
        The target values (class labels in classification, continuous values in regression) for the train samples.
    labels_test : array-like of shape (n_samples,)
        The true target values for the test samples.
    model_parameters : dict
        A dictionary of options to pass to the model. If 'seed' is provided, it will be converted to 'random_state'.

    Returns:
    --------
    tuple
        A tuple containing:
        - model : object
            The trained model object.
        - metrics : dict
            A dictionary of performance metrics computed on the test data.
        - extras : dict
            Additional information including the type of model, model options, and a base score.

    Raises:
    -------
    NotImplementedError
        If the specified model type and output combination is not implemented for Scikit-Learn.
    """   
    def fit_and_predict(self, instances_train, instances_test: numpy.ndarray|pandas.DataFrame, 
                        labels_train, labels_test: numpy.ndarray|pandas.Series,
                        model_parameters: dict):
        model = Scikitlearn.models[self.problem_type][self.model_type]
        if model is None: 
            raise NotImplementedError(str(self.model_type) + " and " + str(self.problem_type) + " is not implemented for ScikitLearn.")

        if 'verbose' in model_parameters.keys() and model_parameters['verbose']<0:
            model_parameters.pop('verbose')
        self.display_parameters(model_parameters)
        
        ARCHITECTURE_KEYS = inspect.signature(model).parameters.keys()
        FIT_KEYS = inspect.signature(model.fit).parameters.keys()
        architecture_parameters = {key:model_parameters[key] for key in model_parameters if key in ARCHITECTURE_KEYS}
        fit_parameters = {key:model_parameters[key] for key in model_parameters if key in FIT_KEYS}
        for key in model_parameters.keys():
            if key not in set(architecture_parameters.keys()).union(set(fit_parameters.keys())):
                print('{k} given in model_parameters but not used'.format(k=key))

        if type(instances_train)==pandas.DataFrame and "subset" in instances_train.columns:
            instances_train = instances_train.drop(columns="subset")
        if type(instances_test)==pandas.DataFrame and "subset" in instances_test.columns:
            instances_test = instances_test.drop(columns="subset")
        instances_train = self._update_instances(instances_train, self.get_item_function)
            
        model = model(**architecture_parameters)
        model.fit(instances_train, labels_train, **fit_parameters)

        result = self.predict(model, instances_test)
        metrics = self.compute_metrics(labels_test, result)
        extras = {
            "model": str(type(model)),
            "model_parameters": model_parameters,
            "base_score": 0,
        }
        return deepcopy(model), metrics, extras


        
    """
    Convert the model to a specific model type and output format.

    This method converts the current model to a specified model type and output format.
    It raises a `NotImplementedError` if the specified combination of model type and output
    is not supported by ScikitLearn.

    Parameters:
    -----------
    problem_type : str|ProblemType
        The type of model to convert the model to. This should be a key in the `models` dictionary.
    output : ModelType
        The output format to convert the model to. This should be a member of the `ModelType` enum.
    learner_information : dict, optional
        Additional information about the model. If provided, this information will be stored in the
        `learner_information` attribute of the instance.

    Returns:
    --------
    model : object
        The converted model in the specified model type and output format.

    Raises:
    -------
    NotImplementedError
        If the specified combination of `problem_type` and `output` is not implemented for ScikitLearn.
    """
    def convert_model(self, learner_information_lst=None):
        model = Scikitlearn.models[self.problem_type][self.model_type]
        if model is None: 
            raise NotImplementedError(str(self.model_type) + " and " + str(self.problem_type) + "is not implemented for ScikitLearn")
              
        if learner_information_lst is not None:
            self.learner_information_lst = learner_information_lst
        if self.model_type == ModelType.DecisionTree:
            return self._to_decision_tree()
        elif self.model_type == ModelType.RandomForest:
            return self._to_random_forests()
        elif self.model_type == ModelType.LinearRidge:
            return self._to_linear_ridge()
        elif self.model_type == ModelType.LinearSimple:
            return self._to_linear_simple()
        elif self.model_type == ModelType.LinearLogistic:
            return self._to_linear_logistic()
        else:
            raise NotImplementedError(str(self.model_type) + " and " + str(self.problem_type) + "is not implemented for ScikitLearn")
    


    def _to_linear_logistic(self):
        models = []
        for id_solver_results, _ in enumerate(self.learner_information_lst):
            model = self.learner_information_lst[id_solver_results].raw_model
            if self.problem_type == ProblemType.Regression:
                raise NotImplementedError(str(self.model_type) + " and " + str(self.problem_type) + "is not implemented for ScikitLearn")
            elif self.problem_type == ProblemType.Classification:
                models.append(LinearModelLogistic(model.n_features_in_, 
                                      coefficients=model.coef_,
                                      intercepts=model.intercept_,
                                      learner_information=self.learner_information_lst[id_solver_results],
                                      feature_names=self.feature_names,
                                      class_names=model.classes_))
        return models
    
    def _to_linear_simple(self):
        models = []
        for id_solver_results, _ in enumerate(self.learner_information_lst):
            model = self.learner_information_lst[id_solver_results].raw_model
            if self.problem_type == ProblemType.Regression:
                models.append(LinearModel(model.n_features_in_, 
                                      coefficients=model.coef_,
                                      intercepts=model.intercept_,
                                      learner_information=self.learner_information_lst[id_solver_results]))
            elif self.problem_type == ProblemType.Classification:
                raise NotImplementedError(str(self.model_type) + " and " + str(self.problem_type) + "is not implemented for ScikitLearn")
    
        return models
    
    def _to_linear_ridge(self):
        models = []
        for id_solver_results, _ in enumerate(self.learner_information_lst):
            model = self.learner_information_lst[id_solver_results].raw_model
            if self.problem_type == ProblemType.Regression:
                models.append(LinearModel(model.n_features_in_, 
                                      coefficients=model.coef_,
                                      intercepts=model.intercept_,
                                      learner_information=self.learner_information_lst[id_solver_results],
                                      feature_names=self.feature_names))
            elif self.problem_type == ProblemType.Classification:
                models.append(LinearModelSimple(model.n_features_in_, 
                                      coefficients=model.coef_,
                                      intercepts=model.intercept_,
                                      learner_information=self.learner_information_lst[id_solver_results],
                                      feature_names=self.feature_names,
                                      class_names=model.classes_))
        return models
        
    def _to_decision_tree(self):
        decision_trees = []
        for id_solver_results, _ in enumerate(self.learner_information_lst):
            sk_tree = self.learner_information_lst[id_solver_results].raw_model
            sk_raw_tree = sk_tree.tree_
            decision_trees.append(self._sk_tree_to_tree(sk_tree, sk_raw_tree, id_solver_results))
        return decision_trees

    
    def _to_random_forests(self):
        random_forests = []
        for id_solver_results, _ in enumerate(self.learner_information_lst):
            random_forest = self.learner_information_lst[id_solver_results].raw_model
            decision_trees = []
            for sk_tree in random_forest:
                sk_raw_tree = sk_tree.tree_
                tree = self._sk_tree_to_tree(sk_tree, sk_raw_tree, id_solver_results)
                decision_trees.append(tree)
            random_forests.append(
                RandomForest(decision_trees, n_features=self.n_features, n_classes=len(sk_tree.classes_), learner_information=self.learner_information_lst[id_solver_results], feature_names=self.feature_names, class_names=random_forest.classes_))
        return random_forests

    """
    Convert a specific Scikitlearn's decision tree into a program-specific object called 'DecisionTree'.
    """
    def _sk_tree_to_tree(self, sk_tree, sk_raw_tree, id_solver_results=0):
        """
        Warning: we use here numpy.argmax(sk_raw_tree.value[0][0]) to get the predition of a tree. 
        But sklearn isn't doing it this way.
        - For us, a leaf of a tree represents a class
        - Not for sklearn where a leaf is a list of probability of classes (example: [0.3 0.4 0.6] for 3 classes)
        This can change the prediction between our model and the sklearn model. 
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
        return DecisionTree(sk_tree.n_features_in_, root, learner_information=self.learner_information_lst[id_solver_results], feature_names=self.feature_names, class_names=sk_tree.classes_)

    def _save_model(self, learner_information, filename):
        file = open(filename + ".pkl", 'wb')
        pickle.dump(learner_information.raw_model, file)
        file.close()
        return filename + ".pkl"

    def _load_model(self, model_file, model_parameters):
        model = None
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        return model
