import copy
import json
import os
from numpy.random import RandomState

import lightgbm

from sklearn.metrics import mean_squared_error, mean_absolute_error
from pyxai import Tools

from pyxai.sources.core.structure.boostedTrees import BoostedTrees, BoostedTreesRegression
from pyxai.sources.core.structure.decisionTree import DecisionTree, DecisionNode, LeafNode
from pyxai.sources.core.tools.utils import compute_accuracy
from pyxai.sources.learning.Learner import Learner, LearnerInformation, NoneData

class LightGBM(Learner):
    """
    Load the dataset, rename the attributes and separe the prediction from the data
    """


    def __init__(self, data=NoneData, types=None):
        super().__init__(data, types)
        self.has_to_display_parameters = True

    def display_parameters(self, learner_options):
        if self.has_to_display_parameters is True:
            Tools.verbose("learner_options:", learner_options)
            self.has_to_display_parameters = False

    def get_solver_name(self):
        return str(self.__class__.__name__)


    def fit_and_predict_DT_CLS(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Decision Tree with classification is not implemented for LightGBM.")
        
    def fit_and_predict_RF_CLS(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Random Forest with classification is not implemented for LightGBM.")

    def fit_and_predict_BT_CLS(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Boosted Trees with classification is not implemented for LightGBM.")

    def fit_and_predict_DT_REG(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Decision Tree with regression is not implemented for LightGBM.")
        
    def fit_and_predict_RF_REG(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        raise NotImplementedError("Random Forest with regression is not implemented for LightGBM.")

    def fit_and_predict_BT_REG(self, instances_training, instances_test, labels_training, labels_test, learner_options):
        self.display_parameters(learner_options)
        learner = lightgbm.LGBMRegressor(**learner_options)
        learner.fit(instances_training, labels_training)
        
        result = learner.predict(instances_test)
        metrics = {
            "mean_squared_error": mean_squared_error(labels_test, result),
            "root_mean_squared_error": mean_squared_error(labels_test, result, squared=False),
            "mean_absolute_error": mean_absolute_error(labels_test, result)
        }

        extras = {
            "base_score": 0
        }
        return (copy.deepcopy(learner), metrics, extras)

    def to_DT_CLS(self, learner_information=None):
        raise NotImplementedError("Decision Tree with classification is not implemented for LightGBM.")

    def to_RF_CLS(self, learner_information=None):
        raise NotImplementedError("Random Forest with classification is not implemented for LightGBM.")

    def to_BT_CLS(self, learner_information=None):
        raise NotImplementedError("Boosted Tree with classification is not implemented for LightGBM.")

    def to_DT_REG(self, learner_information=None):
        raise NotImplementedError("Decision Tree with regression is not implemented for LightGBM.")

    def to_RF_REG(self, learner_information=None):
        raise NotImplementedError("Random Forest with regression is not implemented for LightGBM.")

    
    def to_BT_REG(self, learner_information=None):
        if learner_information is not None: self.learner_information = learner_information
        if self.n_features is None:
            self.n_features = learner_information[0].raw_model.n_features_in_
        
        self.id_features = {"f{}".format(i): i for i in range(self.n_features)}
        BTs = [BoostedTreesRegression(self.results_to_trees(id_solver_results), learner_information=learner_information) for
               id_solver_results, learner_information in enumerate(self.learner_information)]
        return BTs


    def save_model(self, learner_information, filename):
        learner_information.raw_model.save_model(filename + ".model")


    def results_to_trees(self, id_solver_results):
        bt = self.learner_information[id_solver_results].raw_model.booster_
        
        bt_json = self.BT_to_JSON(bt)
        

        decision_trees = []
        target_class = 0
        for i, tree_JSON in enumerate(bt_json["tree_info"]):
            tree_JSON = tree_JSON['tree_structure']
            root = self.recuperate_nodes(tree_JSON)
            decision_trees.append(DecisionTree(self.n_features, root, target_class=[target_class], id_solver_results=id_solver_results))
            if self.n_labels > 2:  # Special case for a 2-classes prediction !
                target_class = target_class + 1 if target_class != self.n_labels - 1 else 0
            
        return decision_trees


    def recuperate_nodes(self, tree_JSON):
        if "split_feature" in tree_JSON:
            id_feature = int(tree_JSON["split_feature"])
            threshold = tree_JSON["threshold"]
            if tree_JSON["decision_type"] != '<=':
                raise NotImplementedError()
            decision_node = DecisionNode(int(id_feature + 1), threshold=threshold, left=None, right=None)
            if "left_child" in tree_JSON: # It is the good, right for yes, left for no
                child = tree_JSON["left_child"]
                decision_node.left = LeafNode(float(child["leaf_value"])) if "leaf_value" in child else self.recuperate_nodes(child)
            if "right_child" in tree_JSON:
                child = tree_JSON["right_child"]
                decision_node.right = LeafNode(float(child["leaf_value"])) if "leaf_value" in child else self.recuperate_nodes(child)
            return decision_node
        elif "leaf_value" in tree_JSON:
            # Special case when the tree is just a leaf, this append when no split is realized by the solver, but the weight have to be take into account
            return LeafNode(float(tree_JSON["leaf_value"]))


    def BT_to_JSON(self, BT):
        #save_names = BT.feature_name()
        #BT.feature_name_ = None
        xgboost_JSON = BT.dump_model()
        #BT.feature_name_ = save_names
        return xgboost_JSON


    def load_model(self, model_file):
        classifier = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        classifier.load_model(model_file)
        return classifier
