import copy
import json
#import ubjson
import os
from numpy.random import RandomState
from numpy import array as np_array
from pandas import DataFrame
import xgboost
from sklearn.preprocessing import LabelEncoder
from collections.abc import Callable

from pyxai import Tools
from pyxai.sources.data_structures.boostedTrees import BoostedTrees
from pyxai.sources.data_structures.boostedTreesRegression import BoostedTreesRegression

from pyxai.sources.data_structures.decisionTree import DecisionTree, DecisionNode, LeafNode
from pyxai.sources.learners.learner import Learner, NoneData
from pyxai.sources.type import OperatorCondition, ProblemType, ModelType, InstancesType, LabelsType

class Xgboost(Learner):
    r""" This class is used to create a model using the [Xgboost](https://xgboost.readthedocs.io/) library. 
    All the methods used in this class (```evaluate```, ```get_instances```, ...) are documented in the parent class ```Learner```.

    """

    models = {
        ProblemType.Classification: {
            ModelType.DecisionTree: None,
            ModelType.RandomForest: None,
            ModelType.BoostedTree: xgboost.XGBClassifier,
        },
        ProblemType.Regression: {
            ModelType.DecisionTree: None,
            ModelType.RandomForest: None,
            ModelType.BoostedTree: xgboost.XGBRegressor,
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
            ```Xgboost```:
               A Learner object.

            Examples
            --------
            >>> from pyxai import Learning, Tools
            >>> learner = Learning.Xgboost(Tools.Options.dataset, problem_type=Learning.CLASSIFICATION)
            >>> model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT, splitting_parameters={'test_size':0.2}, model_parameters={'max_depth':6, 'base_score':0.5})
            
            See Also
            --------
            - [Documentation page](http://localhost:4000/pyxai/documentation/learning/generating/)
        """
        super().__init__(data, problem_type=problem_type, instances_type=instances_type, labels_type=labels_type, model_type=model_type,
                         get_item_function=get_item_function, instances_directory=instances_directory, labels_directory=labels_directory)
        self.has_to_display_parameters = True
        self.label_encoder = None


    def display_parameters(self, model_parameters):
        if self.has_to_display_parameters is True:
            Tools.verbose("model_parameters:", model_parameters)
            self.has_to_display_parameters = False


    @staticmethod
    def get_problem_types():
        return {type(xgboost.XGBClassifier()): (ProblemType.Classification, ModelType.BoostedTree),
                type(xgboost.XGBRegressor()): (ProblemType.Regression, ModelType.BoostedTree)}


    @staticmethod
    def get_learner_name():
        return str(Xgboost.__name__)

    def get_base_score(self, model):
        return model.intercept_        
    
    def fit_and_predict(self, instances_train, instances_test, labels_train, labels_test, model_parameters):
        
        model = Xgboost.models[self.problem_type][self.model_type]
        if model is None: 
            raise NotImplementedError(str(self.model_type) + " and " + str(self.problem_type) + "is not implemented for Xgboost.")
        
        if type(instances_train)==DataFrame and "subset" in instances_train.columns:
            instances_train = instances_train.drop(columns="subset")
        if type(instances_test)==DataFrame and "subset" in instances_test.columns:
            instances_test = instances_test.drop(columns="subset")
        instances_train = self._update_instances(instances_train, self.get_item_function)
            
        self.display_parameters(model_parameters)
        model = model(**model_parameters)
        if self.problem_type == ProblemType.Classification:
            self.label_encoder = LabelEncoder()
            labels_train = self.label_encoder.fit_transform(labels_train)
            self.classes = self.label_encoder.classes_
        model.fit(instances_train, labels_train)
        result = self.predict(model, instances_test)
        metrics = self.compute_metrics(labels_test, result)

        extras = {
            "model": str(type(model)),
            "model_parameters": model_parameters,
            "base_score": self.get_base_score(model),
        }
        return (copy.deepcopy(model), metrics, extras)


    def predict(self, model, instances):
        instances = instances.copy()
        if isinstance(instances, DataFrame) and 'subset' in instances.columns:
            instances = instances.drop(columns='subset')
        instances = self._update_instances(instances, self.get_item_function)
        
        pred = model.predict(instances)
        if self.problem_type == ProblemType.Classification:
            if self.label_encoder is not None:
                pred = self.label_encoder.inverse_transform(pred)
            elif self.classes is not None:
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np_array(self.classes)
                pred = self.label_encoder.inverse_transform(pred)
                
        elif self.labels_type==LabelsType.DiscreteValues:
            pred = pred.round()
        return pred

    
    def convert_model(self, learner_information_lst=None):
        model = Xgboost.models[self.problem_type][self.model_type]
        if model is None: 
            raise NotImplementedError(str(self.model_type) + " and " + str(self.problem_type) + "is not implemented for Xgboost.")
              
        if learner_information_lst is not None:
            self.learner_information_lst = learner_information_lst
        if self.n_features is None:
            self.n_features = learner_information_lst[0].raw_model.n_features_in_
        self.id_features = {"f{}".format(i): i for i in range(self.n_features)}

        if self.problem_type == ProblemType.Classification:
            if self.n_labels is None:
                self.n_labels = len(self.learner_information_lst[0].raw_model.classes_)
            return [BoostedTrees(self.to_boosted_trees(id_solver_results), n_classes=self.n_labels, learner_information=learner_information, class_names=self.classes, feature_names=self.feature_names) for
               id_solver_results, learner_information in enumerate(self.learner_information_lst)]
        elif self.problem_type == ProblemType.Regression:
            return [BoostedTreesRegression(self.to_boosted_trees(id_solver_results), learner_information=learner_information, feature_names=self.feature_names) for
               id_solver_results, learner_information in enumerate(self.learner_information_lst)]
        

    def to_boosted_trees(self, id_solver_results):
        xgb_BT = self.learner_information_lst[id_solver_results].raw_model.get_booster()
        xgb_JSON = self._xgboost_BT_to_JSON(xgb_BT)
        base_score = json.loads(xgb_JSON['learner']['learner_model_param']['base_score'])
        if len(base_score)==1: base_score=base_score[0]
        self.learner_information_lst[id_solver_results].extras["base_score"] = base_score
        if self.problem_type == ProblemType.Classification and self.classes is None: self.classes = self.learner_information_lst[id_solver_results].raw_model.classes_
        elif self.problem_type == ProblemType.Regression: self.classes = None
        decision_trees = []
        associated_class = 0 
        for tree_JSON in xgb_JSON['learner']['gradient_booster']['model']['trees']:
            root = self._recuperate_nodes(tree_JSON)

            decision_tree = DecisionTree(self.n_features, root, learner_information=self.learner_information_lst[id_solver_results], class_names=self.classes)
            decision_tree.set_associated_class(associated_class)
            decision_trees.append(decision_tree)
            
            if self.problem_type == ProblemType.Classification and self.n_labels > 2:  # Special case for a 2-classes prediction !
                associated_class = associated_class + 1 if associated_class != self.n_labels - 1 else 0
            
        return decision_trees


    def _recuperate_nodes(self, tree_JSON, id=0):        
        if tree_JSON['left_children'][id] == -1 and tree_JSON['right_children'][id] == -1:
            return LeafNode(tree_JSON['split_conditions'][id])

        id_feature = tree_JSON['split_indices'][id]
        threshold = tree_JSON["split_conditions"][id]
        type_condition = tree_JSON["split_type"][id]
        if type_condition == 0:
            operator = OperatorCondition.GE
        else:
            raise NotImplementedError("The type of split in the tree have to be implemented for the Learner, this is probably due to using categorical feateures without encoding.")
        
        decision_node = DecisionNode(int(id_feature) + 1, operator=operator, threshold=threshold, left=None, right=None)
        decision_node.right = self._recuperate_nodes(tree_JSON, tree_JSON["right_children"][id])
        decision_node.left = self._recuperate_nodes(tree_JSON, tree_JSON["left_children"][id])
        return decision_node    
        
        # exit(0) # TODO Remove (12/02/2026)
        # if "children" in tree_JSON:
        #     assert tree_JSON["split"] in self.id_features, "A feature is not correct during the parsing from xgb_JSON to DT !"
        #     id_feature = self.id_features[tree_JSON["split"]]
        #     threshold = tree_JSON["split_condition"]
        #     decision_node = DecisionNode(int(id_feature + 1), operator=OperatorCondition.GE, threshold=threshold, left=None, right=None)
        #     id_right = tree_JSON["no"]  # It is the inverse here, right for no, left for yes
        #     for child in tree_JSON["children"]:
        #         if child["nodeid"] == id_right:
        #             decision_node.right = self.recuperate_nodes(child)
        #         else:
        #             decision_node.left = self.recuperate_nodes(child)
                    
        #     return decision_node
        # elif "leaf" in tree_JSON:
        #     # Special case when the tree is just a leaf, this append when no split is realized by the solver, but the weight have to be take into account
        #     return LeafNode(tree_JSON["leaf"])


    def _xgboost_BT_to_JSON(self, xgboost_BT):
        xgboost_BT.save_model("xgboost_JSON.json")
        xgboost_JSON = None
        with open('xgboost_JSON.json', "r") as f:
            xgboost_JSON = json.load(f)
        os.remove("xgboost_JSON.json")
        return xgboost_JSON

    def _save_model(self, learner_information, filename):
        learner_information.raw_model.save_model(filename + ".ubj")
        return filename + ".ubj"
        
    def _load_model(self, model_file, model_parameters):
        if self.problem_type == ProblemType.Classification:
            model = xgboost.XGBClassifier(**model_parameters)
        elif self.problem_type == ProblemType.Regression:
            model = xgboost.XGBRegressor(**model_parameters)
        else:
            raise ValueError("problem_type is unknown:", str(self.problem_type))
        model.load_model(model_file)  # xgboost library function
        return model

    