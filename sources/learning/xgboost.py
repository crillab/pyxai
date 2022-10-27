import copy
import json
import os
from numpy.random import RandomState

import xgboost
from pyxai.sources.core.structure.boostedTrees import BoostedTrees
from pyxai.sources.core.structure.decisionTree import DecisionTree, DecisionNode, LeafNode
from pyxai.sources.core.tools.utils import compute_accuracy
from pyxai.sources.learning.Learner import Learner, LearnerInformation, NoneData


class Xgboost(Learner):
    """
    Load the dataset, rename the attributes and separe the prediction from the data
    """


    def __init__(self, data=NoneData):
        super().__init__(data)


    def get_solver_name(self):
        return str(self.__class__.__name__)


    def fit_and_predict_DT(self, instances_training, instances_test, labels_training, labels_test, max_depth=None, seed=0):
        assert False, "Xgboost is not able to produce DT !"


    def fit_and_predict_RF(self, instances_training, instances_test, labels_training, labels_test, max_depth=None, seed=0):
        assert False, "Xgboost is not able to produce RF !"


    def fit_and_predict_BT(self, instances_training, instances_test, labels_training, labels_test, max_depth=None, seed=0):
        # Training phase
        if seed is None: seed = RandomState(None)
        xgb_classifier = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=max_depth, random_state=seed)
        xgb_classifier.fit(instances_training, labels_training)
        # Test phase
        result = xgb_classifier.predict(instances_test)
        return (copy.deepcopy(xgb_classifier), compute_accuracy(result, labels_test))


    def to_DT(self, learner_information=None):
        assert True, "Xgboost is only able to evaluate a classifier in the form of boosted trees"


    def to_RF(self, learner_information=None):
        assert True, "Xgboost is only able to evaluate a classifier in the form of boosted trees"


    def to_BT(self, learner_information=None):
        if learner_information is not None: self.learner_information = learner_information
        if self.n_features is None:
            self.n_features = learner_information[0].raw_model.n_features_in_
        if self.n_labels is None:
            self.n_labels = len(learner_information[0].raw_model.classes_)

        self.id_features = {"f{}".format(i): i for i in range(self.n_features)}
        BTs = [BoostedTrees(self.results_to_trees(id_solver_results), n_classes=self.n_labels, learner_information=learner_information) for
               id_solver_results, learner_information in enumerate(self.learner_information)]
        return BTs


    def save_model(self, learner_information, filename):
        learner_information.raw_model.save_model(filename + ".model")


    def results_to_trees(self, id_solver_results):
        xgb_BT = self.learner_information[id_solver_results].raw_model.get_booster()
        xgb_JSON = self.xgboost_BT_to_JSON(xgb_BT)
        decision_trees = []
        target_class = 0
        for _, tree_JSON in enumerate(xgb_JSON):
            tree_JSON = json.loads(tree_JSON)
            root = self.recuperate_nodes(tree_JSON)
            decision_trees.append(DecisionTree(self.n_features, root, target_class=[target_class], id_solver_results=id_solver_results))
            if self.n_labels > 2:  # Special case for a 2-classes prediction !
                target_class = target_class + 1 if target_class != self.n_labels - 1 else 0
        return decision_trees


    def recuperate_nodes(self, tree_JSON):
        if "children" in tree_JSON:
            assert tree_JSON["split"] in self.id_features, "A feature is not correct during the parsing from xgb_JSON to DT !"
            id_feature = self.id_features[tree_JSON["split"]]
            # print("id_features:", id_feature)
            threshold = tree_JSON["split_condition"]
            decision_node = DecisionNode(int(id_feature + 1), threshold=threshold, left=None, right=None)
            id_right = tree_JSON["no"]  # It is the inverse here, right for no, left for yes
            for child in tree_JSON["children"]:
                if child["nodeid"] == id_right:
                    decision_node.right = LeafNode(float(child["leaf"])) if "leaf" in child else self.recuperate_nodes(child)
                else:
                    decision_node.left = LeafNode(float(child["leaf"])) if "leaf" in child else self.recuperate_nodes(child)
            return decision_node
        elif "leaf" in tree_JSON:
            # Special case when the tree is just a leaf, this append when no split is realized by the solver, but the weight have to be take into account
            return LeafNode(float(tree_JSON["leaf"]))


    def xgboost_BT_to_JSON(self, xgboost_BT):
        save_names = xgboost_BT.feature_names
        xgboost_BT.feature_names = None
        xgboost_JSON = xgboost_BT.get_dump(with_stats=True, dump_format="json")
        xgboost_BT.feature_names = save_names
        return xgboost_JSON


    def load_model(self, model_file):
        classifier = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        classifier.load_model(model_file)
        return classifier
