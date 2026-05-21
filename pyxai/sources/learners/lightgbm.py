import copy
import json
import os
import pickle
from numpy.random import RandomState
from pandas import DataFrame
from collections.abc import Callable

import lightgbm

from sklearn.metrics import mean_squared_error, mean_absolute_error
from pyxai import Tools


from pyxai.sources.data_structures.boostedTreesRegression import BoostedTreesRegression

from pyxai.sources.data_structures.decisionTree import DecisionTree, DecisionNode, LeafNode
from pyxai.sources.learners.learner import Learner, NoneData
from pyxai.sources.type import ProblemType, ModelType, InstancesType, LabelsType

class LightGBM(Learner):
    r""" This class is used to create a model using the [LightGBM](https://lightgbm.readthedocs.io/en/stable/) library. 
    All the methods used in this class (```evaluate```, ```get_instances```, ...) are documented in the parent class ```Learner```.
    """
    
    models = {
        ProblemType.Classification: {
            ModelType.DecisionTree: None,
            ModelType.RandomForest: None,
            ModelType.BoostedTree: None,
        },
        ProblemType.Regression: {
            ModelType.DecisionTree: None,
            ModelType.RandomForest: None,
            ModelType.BoostedTree: lightgbm.LGBMRegressor,
        },
    }

    # /!\ Docstring of this method come from the parent class Learner. 
    def __init__(self, data: str|DataFrame|NoneData = NoneData, *,
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
            ```LightGBM```:
               A Learner object.

            Examples
            --------
            >>> from pyxai import Learning, Tools
            >>> learner = Learning.LightGBM(Tools.Options.dataset, problem_type=Learning.CLASSIFICATION)
            >>> model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT, splitting_parameters={'test_size':0.2}, model_parameters={'max_depth':6, 'base_score':0.5})
            
            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/learning/generating/)
        """
    
        super().__init__(data, problem_type=problem_type, instances_type=instances_type, labels_type=labels_type, model_type=model_type,
                         get_item_function=get_item_function, instances_directory=instances_directory, labels_directory=labels_directory)

        self.has_to_display_parameters = True
        

    def display_parameters(self, model_parameters):
        if self.has_to_display_parameters is True:
            Tools.verbose("model_parameters:", model_parameters)
            self.has_to_display_parameters = False

            
    @staticmethod
    def get_problem_types():
        return {type(lightgbm.LGBMRegressor(verbose=-1)): (ProblemType.Regression, ModelType.BoostedTree)}

    
    @staticmethod
    def get_learner_name():
        return str(LightGBM.__name__)

        
    def fit_and_predict(self, instances_train, instances_test, labels_train, labels_test, model_parameters):
        model = LightGBM.models[self.problem_type][self.model_type]
        if model is None: 
            raise NotImplementedError(str(self.model_type) + " and " + str(self.problem_type) + "is not implemented for LightGBM.")
        
        if type(instances_train)==DataFrame and "subset" in instances_train.columns:
            instances_train = instances_train.drop(columns="subset")
        if type(instances_test)==DataFrame and "subset" in instances_test.columns:
            instances_test = instances_test.drop(columns="subset")
        instances_train = self._update_instances(instances_train, self.get_item_function)

        if "verbose" not in model_parameters.keys():
            model_parameters["verbose"] = -1

        self.display_parameters(model_parameters)
        model = model(**model_parameters)
        model.fit(instances_train, labels_train)
        result = self.predict(model, instances_test)
        metrics = self.compute_metrics(labels_test, result)
        
        extras = {
            "model": str(type(model)),
            "model_parameters": model_parameters,
            "base_score": 0,
        }
        return (copy.deepcopy(model), metrics, extras)

        
    def convert_model(self, learner_information_lst=None):
        model = LightGBM.models[self.problem_type][self.model_type]
        if model is None: 
            raise NotImplementedError(str(self.model_type) + " and " + str(self.problem_type) + "is not implemented for LightGBM.")
              
        if learner_information_lst is not None:
            self.learner_information_lst = learner_information_lst
        
        if self.n_features is None:
            self.n_features = learner_information_lst[0].raw_model.n_features_in_
        self.id_features = {"f{}".format(i): i for i in range(self.n_features)}

        return [BoostedTreesRegression(self._to_boosted_trees(id_solver_results), learner_information=learner_information, feature_names=self.feature_names) for
               id_solver_results, learner_information in enumerate(self.learner_information_lst)]

    
    def _to_boosted_trees(self, id_solver_results):
        bt = self.learner_information_lst[id_solver_results].raw_model.booster_
        
        bt_json = bt.dump_model()
        decision_trees = []
        for i, tree_JSON in enumerate(bt_json["tree_info"]):
            tree_JSON = tree_JSON['tree_structure']
            root = self._recuperate_nodes(tree_JSON)
            decision_trees.append(DecisionTree(self.n_features, root, learner_information=self.learner_information_lst[id_solver_results], class_names=None))
            
        return decision_trees


    def _recuperate_nodes(self, tree_JSON):
        if "split_feature" in tree_JSON:
            id_feature = int(tree_JSON["split_feature"])
            threshold = tree_JSON["threshold"]
            if tree_JSON["decision_type"] != '<=':
                raise NotImplementedError()
            decision_node = DecisionNode(int(id_feature + 1), threshold=threshold, left=None, right=None)
            if "left_child" in tree_JSON: # It is the good, right for yes, left for no
                child = tree_JSON["left_child"]
                decision_node.left = LeafNode(float(child["leaf_value"])) if "leaf_value" in child else self._recuperate_nodes(child)
            if "right_child" in tree_JSON:
                child = tree_JSON["right_child"]
                decision_node.right = LeafNode(float(child["leaf_value"])) if "leaf_value" in child else self._recuperate_nodes(child)
            return decision_node
        elif "leaf_value" in tree_JSON:
            # Special case when the tree is just a leaf, this append when no split is realized by the solver, but the weight have to be take into account
            return LeafNode(float(tree_JSON["leaf_value"]))

    
    def _save_model(self, learner_information, filename):
        #learner_information.raw_model.booster_.save_model(filename + ".model")
        file = open(filename + ".pkl", 'wb')
        pickle.dump(learner_information.raw_model, file)
        file.close()
        return filename + ".pkl"

    
    def _load_model(self, model_file, model_parameters):
        #model = lightgbm.Booster(model_file=model_file)
        model = None
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        return model
        