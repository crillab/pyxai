import copy
import json
import os
from numpy.random import RandomState

import xgboost
import numpy
from pyxai import Tools
from pyxai.sources.core.structure.boostedTrees import BoostedTrees, BoostedTreesRegression
from pyxai.sources.core.structure.randomForest import RandomForest
from pyxai.sources.core.structure.decisionTree import DecisionTree, DecisionNode, LeafNode
from pyxai.sources.learning.learner import Learner, NoneData
from pyxai.sources.core.structure.type import OperatorCondition, LearnerType, EvaluationOutput

class Xgboost(Learner):
    """
    Load the dataset, rename the attributes and separe the prediction from the data
    """

    learners = {
        LearnerType.Classification: {
            EvaluationOutput.DT: None,
            EvaluationOutput.RF: None,
            EvaluationOutput.BT: xgboost.XGBClassifier,
        },
        LearnerType.Regression: {
            EvaluationOutput.DT: None,
            EvaluationOutput.RF: None,
            EvaluationOutput.BT: xgboost.XGBRegressor,
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
        return {type(xgboost.XGBClassifier()): (LearnerType.Classification, EvaluationOutput.BT),
                type(xgboost.XGBRegressor()): (LearnerType.Regression, EvaluationOutput.BT)}

    @staticmethod
    def get_learner_name():
        return str(Xgboost.__name__)
    
    def fit_and_predict(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        learner = Xgboost.learners[self.learner_type][self.models_type]
        if learner is None: 
            raise NotImplementedError(str(self.models_type) + " and " + str(self.learner_type) + "is not implemented for Xgboost.")
        
        if "eval_metric" not in learner_options.keys():
            learner_options["eval_metric"] = "mlogloss"
        
        self.display_parameters(learner_options)
        learner = learner(**learner_options)
        learner.fit(instances_training, labels_training)
       
        result = learner.predict(instances_test)
        metrics = self.compute_metrics(labels_test, result)
        
        # Special case for Regression with a base_score used to compute predictions
        if self.learner_type == LearnerType.Regression:
            base_score = float(0.5) if learner.base_score is None else learner.base_score
        elif self.learner_type == LearnerType.Classification:
            base_score = 0
        
        extras = {
            "learner": str(type(learner)),
            "learner_options": learner_options,
            "base_score": base_score,
        }
        return (copy.deepcopy(learner), metrics, extras)
    
    def convert_model(self, learner_information=None):
        learner = Xgboost.learners[self.learner_type][self.models_type]
        if learner is None: 
            raise NotImplementedError(str(self.models_type) + " and " + str(self.learner_type) + "is not implemented for Xgboost.")
              
        if learner_information is not None:
            self.learner_information = learner_information 
        
        if self.n_features is None:
            self.n_features = learner_information[0].raw_model.n_features_in_
        self.id_features = {"f{}".format(i): i for i in range(self.n_features)}

        if self.learner_type == LearnerType.Classification:
            if self.n_labels is None:
                self.n_labels = len(learner_information[0].raw_model.classes_)
            return [BoostedTrees(self.to_boosted_trees(id_solver_results), n_classes=self.n_labels, learner_information=learner_information) for
               id_solver_results, learner_information in enumerate(self.learner_information)]
        elif self.learner_type == LearnerType.Regression:
            return [BoostedTreesRegression(self.to_boosted_trees(id_solver_results), learner_information=learner_information) for
               id_solver_results, learner_information in enumerate(self.learner_information)]
        

    def to_boosted_trees(self, id_solver_results):
        xgb_BT = self.learner_information[id_solver_results].raw_model.get_booster()
        xgb_JSON = self.xgboost_BT_to_JSON(xgb_BT)
        decision_trees = []
        target_class = 0
        for tree_JSON in xgb_JSON:
            tree_JSON = json.loads(tree_JSON)
            root = self.recuperate_nodes(tree_JSON)
            
            decision_trees.append(DecisionTree(self.n_features, root, target_class=[target_class], id_solver_results=id_solver_results))
            if self.learner_type == LearnerType.Classification and self.n_labels > 2:  # Special case for a 2-classes prediction !
                target_class = target_class + 1 if target_class != self.n_labels - 1 else 0
            
        return decision_trees

    def recuperate_nodes(self, tree_JSON):
        if "children" in tree_JSON:
            assert tree_JSON["split"] in self.id_features, "A feature is not correct during the parsing from xgb_JSON to DT !"
            id_feature = self.id_features[tree_JSON["split"]]
            threshold = tree_JSON["split_condition"]
            decision_node = DecisionNode(int(id_feature + 1), operator=OperatorCondition.GT, threshold=threshold, left=None, right=None)
            id_right = tree_JSON["no"]  # It is the inverse here, right for no, left for yes
            for child in tree_JSON["children"]:
                if child["nodeid"] == id_right:
                    decision_node.right = self.recuperate_nodes(child)
                else:
                    decision_node.left = self.recuperate_nodes(child)
                    
            return decision_node
        elif "leaf" in tree_JSON:
            # Special case when the tree is just a leaf, this append when no split is realized by the solver, but the weight have to be take into account
            return LeafNode(tree_JSON["leaf"])


    def xgboost_BT_to_JSON(self, xgboost_BT):
        save_names = xgboost_BT.feature_names
        xgboost_BT.feature_names = None
        xgboost_JSON = xgboost_BT.get_dump(with_stats=True, dump_format="json")
        xgboost_BT.feature_names = save_names
        return xgboost_JSON

    def save_model(self, learner_information, filename):
        learner_information.raw_model.save_model(filename + ".model")
        
    def load_model(self, model_file, learner_options):
        if self.learner_type == LearnerType.Classification:
            learner = xgboost.XGBClassifier(**learner_options)
        elif self.learner_type == LearnerType.Regression:
            learner = xgboost.XGBRegressor(**learner_options)
        else:
            raise ValueError("learner_type is unknown:", str(self.learner_type))
        learner.load_model(model_file)
        return learner

    