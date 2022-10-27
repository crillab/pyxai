import json

from pyxai.sources.core.structure.boostedTrees import BoostedTrees
from pyxai.sources.core.structure.decisionTree import DecisionTree, DecisionNode
from pyxai.sources.core.structure.randomForest import RandomForest
from pyxai.sources.learning.Learner import Learner, NoneData


class Generic(Learner):
    def __init__(self, data=NoneData):
        super().__init__(data)


    def get_solver_name(self):
        return str(self.__class__.__name__)


    def fit_and_predict_DT(self, instances_training, instances_test, labels_training, labels_test):
        assert False, "No possible evaluation for a generic learner"


    def fit_and_predict_RF(self, instances_training, instances_test, labels_training, labels_test):
        assert False, "No possible evaluation for a generic learner"


    def fit_and_predict_BT(self, instances_training, instances_test, labels_training, labels_test):
        assert False, "No possible evaluation for a generic learner"


    """
    Convert the Scikitlearn's decision trees into the program-specific objects called 'DecisionTree'.
    """


    def to_DT(self, learner_information=None):
        if learner_information is not None: self.learner_information = learner_information
        decision_trees = []
        for id_solver_results, _ in enumerate(self.learner_information):
            dt = self.learner_information[id_solver_results].raw_model
            decision_trees.append(self.classifier_to_DT(dt, id_solver_results))
        return decision_trees


    def to_RF(self, learner_information=None):
        if learner_information is not None: self.learner_information = learner_information
        random_forests = []
        for id_solver_results, _ in enumerate(self.learner_information):
            random_forest = self.learner_information[id_solver_results].raw_model
            n_classes = random_forest[0]
            decision_trees = []
            for dt in random_forest[1]:
                decision_trees.append(self.classifier_to_DT(dt, id_solver_results))
            random_forests.append(
                RandomForest(decision_trees, n_classes=n_classes, learner_information=self.learner_information[id_solver_results]))
        return random_forests


    def to_BT(self, learner_information=None):
        if learner_information is not None: self.learner_information = learner_information
        boosted_trees = []
        for id_solver_results, _ in enumerate(self.learner_information):
            random_forest = self.learner_information[id_solver_results].raw_model
            n_classes = random_forest[0]
            decision_trees = []
            for dt in random_forest[1]:
                decision_trees.append(self.classifier_to_DT(dt, id_solver_results))
            boosted_trees.append(BoostedTrees(decision_trees, n_classes=n_classes, learner_information=self.learner_information[id_solver_results]))
        return boosted_trees


    def classifier_to_DT(self, raw_dt, id_solver_results=0):
        n_features = raw_dt[0]
        target_class = raw_dt[1]
        root = self.raw_node_to_decision_node(raw_dt[2])
        return DecisionTree(n_features, root, target_class, id_solver_results=id_solver_results,
                            learner_information=self.learner_information[id_solver_results])


    def raw_node_to_decision_node(self, raw_dt):
        if isinstance(raw_dt, (float, int)):
            # Leaf
            return raw_dt
        elif isinstance(raw_dt[0], str):
            # Node
            assert raw_dt[0].startswith("f"), "Have to start by f !"
            id_feature = int(raw_dt[0].split("<")[0].split("f")[1].strip())
            threshold = float(raw_dt[0].split("<")[1].strip())
            return DecisionNode(id_feature, threshold=threshold, left=self.raw_node_to_decision_node(raw_dt[1]),
                                right=self.raw_node_to_decision_node(raw_dt[2]))
        else:
            assert False, "It is not possible !"

    @staticmethod
    def load_model(model_file):
        f = open(model_file)
        classifier = json.loads(json.load(f))
        f.close()
        return classifier
