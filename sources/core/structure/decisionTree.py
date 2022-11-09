import copy
import numpy
import os

from pyxai.sources.core.structure.binaryMapping import BinaryMapping
from pyxai.sources.core.structure.decisionNode import DecisionNode, LeafNode
from pyxai.sources.core.structure.type import TypeLeaf, Encoding, OperatorCondition
from pyxai.sources.core.tools.encoding import CNFencoding


class DecisionTree(BinaryMapping):

    def __init__(self, n_features, root, target_class=0, id_solver_results=0, learner_information=None, force_features_equal_to_binaries=False):
        """

        Args:
            n_features (_type_): _description_
            root (_type_): _description_
            target_class (int, optional): _description_. Defaults to 0.
            id_solver_results (int, optional): _description_. Defaults to 0.
            learner_information (_type_, optional): _description_. Defaults to None.
            force_features_equal_to_binaries (bool, optional): By default, the binaries id in the representation of implicants
            is not the same that the features id. This option allows to force that these two kinds of ids are the same. This is used to allows
            that variables directly represent the features. Defaults to False.
        """
        self.id_solver_results = id_solver_results
        self.learner_information = learner_information
        self.n_features = n_features
        self.nodes = []
        self.root = root
        self.target_class = target_class  # can be a integer (for BT) or a list (for DT and RF) TODO
        if not self.root.is_leaf():
            self.define_parents(self.root)
        self.force_features_equal_to_binaries = force_features_equal_to_binaries
        self.map_id_binaries_to_features, self.map_features_to_id_binaries = self.compute_id_binaries(force_features_equal_to_binaries)
        super().__init__(self.map_id_binaries_to_features, self.map_features_to_id_binaries, learner_information)

        # assert isinstance(self.type_tree, TypeTree), "Please put the good type of the tree !"


    def __str__(self):
        s = "**Decision Tree Model**" + os.linesep
        s += "nFeatures: " + str(self.n_features) + os.linesep
        s += "nNodes: " + str(len(self.nodes)) + os.linesep
        s += "nVariables: " + str(len(self.map_id_binaries_to_features) - 1) + os.linesep
        return s


    def raw_data_for_CPP(self):
        raw_t = tuple([self.root.value]) if self.root.is_leaf() else self.to_tuples(self.root, for_cpp=True)
        return (int(self.target_class[0]) if isinstance(self.target_class, (numpy.ndarray, list, tuple)) else self.target_class, raw_t)


    def raw_data(self):
        raw = tuple([self.root.value]) if self.root.is_leaf() else self.to_tuples(self.root)
        return (int(self.n_features), [int(element) for element in self.target_class], raw)


    def to_tuples(self, node, for_cpp=False):
        """
        For example, this method can return (1, (2, (2.5,3.5)), (3 (-1.5, 0.5)))
        for a tree with 3 nodes and the leaves with the weights 2.5 3.5 -1.5 0.5.
        """
        output = [str(node)] if for_cpp == False else [self.get_id_variable(node)]
        if not node.left.is_leaf():
            output.append(self.to_tuples(node.left, for_cpp))
        else:
            output.append(node.left.value if not isinstance(node.left.value, numpy.int64) else int(node.left.value))
        if not node.right.is_leaf():
            output.append(self.to_tuples(node.right, for_cpp))
        else:
            output.append(node.right.value if not isinstance(node.right.value, numpy.int64) else int(node.right.value))
        return tuple(output)


    def simplify(self):
        while self._simplify(self.root):
            pass


    def _simplify(self, node, path=[], come_from=None, previous_node=None, previous_previous_node=None):
        res_1 = False
        res_2 = False
        change = False
        if previous_node is not None:
            new_tuple = (self.get_id_variable(previous_node), come_from)
            if new_tuple in path:
                if path[-1][1] == 0:
                    previous_previous_node.left = node
                elif path[-1][1] == 1:
                    previous_previous_node.right = node
                change = True
            path.append(new_tuple)

        # print("path:", path)
        if not node.is_leaf():
            raw = self.to_tuples(node)
            if raw[1] == raw[2]:
                if come_from == 0:
                    previous_node.left = node.left
                if come_from == 1:
                    previous_node.right = node.right
                change = True
            pp = previous_node
            res_1 = self._simplify(node.left, copy.deepcopy(path), come_from=0, previous_node=node, previous_previous_node=pp)
            res_2 = self._simplify(node.right, copy.deepcopy(path), come_from=1, previous_node=node, previous_previous_node=pp)
        return res_1 or res_2 or change


    def negating_tree(self):
        new_tree = copy.deepcopy(self)
        new_tree.root.negating_tree()
        return new_tree


    def concatenate_tree(self, other_tree):
        new_tree = copy.deepcopy(self)
        new_tree.root.concatenate_tree(other_tree)
        new_tree.concatenate_id_binaries(other_tree)
        return new_tree


    def disjoint_tree(self, other_tree):
        new_tree = copy.deepcopy(self)
        new_tree.root.concatenate_tree(other_tree, disjunction=True)
        new_tree.concatenate_id_binaries(other_tree)
        return new_tree


    def get_variables(self, binary_representation=None, node=None):
        if node is None:
            if self.root.is_leaf():
                return []
            node = self.root
        output = []
        if binary_representation is None:
            output.append(self.get_id_variable(node))
        else:
            if self.get_id_variable(node) in binary_representation:
                output.append(self.get_id_variable(node))
            if -self.get_id_variable(node) in binary_representation:
                output.append(-self.get_id_variable(node))

        if not node.left.is_leaf() and not node.right.is_leaf():
            return output + self.get_variables(binary_representation, node.left) + self.get_variables(binary_representation, node.right)
        elif not node.left.is_leaf():
            return output + self.get_variables(binary_representation, node.left)
        elif not node.right.is_leaf():
            return output + self.get_variables(binary_representation, node.right)
        return output


    def get_features(self, node=None):
        if node is None:
            if self.root.is_leaf():
                return []
            node = self.root

        output = [node.id_feature]

        if not node.left.is_leaf() and not node.right.is_leaf():
            return output + self.get_features(node.left) + self.get_features(node.right)
        elif not node.left.is_leaf():
            return output + self.get_features(node.left)
        elif not node.right.is_leaf():
            return output + self.get_features(node.right)
        return output


    def direct_reason(self, instance, node=None):
        if node is None:
            node = self.root
        if node.is_leaf():
            return []
        output = []
        value = instance[node.id_feature - 1]
        if node.operator == OperatorCondition.GE:
            if value >= node.threshold:
                output.append(self.get_id_variable(node))
                return output + self.direct_reason(instance, node.right) if not node.right.is_leaf() else output
            else:
                output.append(-self.get_id_variable(node))
                return output + self.direct_reason(instance, node.left) if not node.left.is_leaf() else output
        elif node.operator == OperatorCondition.GT:
            if value > node.threshold:
                output.append(self.get_id_variable(node))
                return output + self.direct_reason(instance, node.right) if not node.right.is_leaf() else output
            else:
                output.append(-self.get_id_variable(node))
                return output + self.direct_reason(instance, node.left) if not node.left.is_leaf() else output
        elif node.operator == OperatorCondition.LE:
            if value <= node.threshold:
                output.append(self.get_id_variable(node))
                return output + self.direct_reason(instance, node.right) if not node.right.is_leaf() else output
            else:
                output.append(-self.get_id_variable(node))
                return output + self.direct_reason(instance, node.left) if not node.left.is_leaf() else output
        elif node.operator == OperatorCondition.LT:
            if value < node.threshold:
                output.append(self.get_id_variable(node))
                return output + self.direct_reason(instance, node.right) if not node.right.is_leaf() else output
            else:
                output.append(-self.get_id_variable(node))
                return output + self.direct_reason(instance, node.left) if not node.left.is_leaf() else output
        elif node.operator == OperatorCondition.EQ:
            if value == node.threshold:
                output.append(self.get_id_variable(node))
                return output + self.direct_reason(instance, node.right) if not node.right.is_leaf() else output
            else:
                output.append(-self.get_id_variable(node))
                return output + self.direct_reason(instance, node.left) if not node.left.is_leaf() else output
        elif node.operator == OperatorCondition.NEQ:
            if value != node.threshold:
                output.append(self.get_id_variable(node))
                return output + self.direct_reason(instance, node.right) if not node.right.is_leaf() else output
            else:
                output.append(-self.get_id_variable(node))
                return output + self.direct_reason(instance, node.left) if not node.left.is_leaf() else output


    def define_parents(self, node, *, parent=None):
        if not node.is_leaf():
            self.nodes.append(node)
            self.define_parents(node.left, parent=node)
            self.define_parents(node.right, parent=node)
        if parent is not None:
            node.parent = parent


    def concatenate_id_binaries(self, other_tree):
        self.map_features_to_id_binaries.update(other_tree.map_features_to_id_binaries)
        for i, element in enumerate(self.map_id_binaries_to_features):
            if element is None:
                self.map_id_binaries_to_features[i] = other_tree.map_id_binaries_to_features[i]


    def compute_id_binaries(self, force_features_equal_to_binaries=False):
        """
        Overload method from the mother class BinaryMapping
        map_id_binaries_to_features: list[id_binary] -> (id_feature, operator, threshold)
        map_features_to_id_binaries: dict[(id_feature, operator, threshold)] -> [id_binary, n_appears, n_appears_per_tree]
        """
        if not force_features_equal_to_binaries:
            map_id_binaries_to_features = [0]
        else:
            map_id_binaries_to_features = [0] + [None] * self.n_features

        map_features_to_id_binaries = {}
        id_binary = 1
        for node in self.nodes:
            if (node.id_feature, node.operator, node.threshold) not in map_features_to_id_binaries:
                if not force_features_equal_to_binaries:
                    map_features_to_id_binaries[(node.id_feature, node.operator, node.threshold)] = [id_binary, 1, None]
                    map_id_binaries_to_features.append((node.id_feature, node.operator, node.threshold))
                    id_binary += 1
                else:
                    map_features_to_id_binaries[(node.id_feature, node.operator, node.threshold)] = [node.id_feature, 1, None]
                    map_id_binaries_to_features[node.id_feature] = (node.id_feature, node.operator, node.threshold)
            else:
                map_features_to_id_binaries[(node.id_feature, node.operator, node.threshold)][1] += 1
        return (map_id_binaries_to_features, map_features_to_id_binaries)


    def get_id_variable(self, node):
        return self.map_features_to_id_binaries[(node.id_feature, node.operator, node.threshold)][0]


    def is_leaf(self):
        return self.root.is_leaf()


    def compute_nodes_with_leaves(self, node):
        """
        Return a list of tuple representing the children of start_node
        """

        output = []
        if node.left.is_leaf() or node.right.is_leaf():
            output.append(node)
        if not node.left.is_leaf():
            output += self.compute_nodes_with_leaves(node.left)
        if not node.right.is_leaf():
            output += self.compute_nodes_with_leaves(node.right)
        return output


    def display(self, node):
        if node.is_leaf():
            print(node)
        else:
            print(node)
            self.display(node.left)
            self.display(node.right)


    def is_implicant(self, reason, target_prediction):
        return self.root.is_implicant(reason, target_prediction, self.map_features_to_id_binaries)


    def predict_instance(self, instance):
        """
        Return the prediction (the classification) of an observation (instance) according to this tree
        """
        return self.root.take_decisions_instance(instance)


    def take_decisions_binary_representation(self, binary_representation, map_id_binaries_to_features=None):
        """
        Return the prediction (the classification) of an binary representation according to this tree
        """
        if map_id_binaries_to_features is None:
            map_id_binaries_to_features = self.map_id_binaries_to_features
        return self.root.take_decisions_binary_representation(binary_representation, map_id_binaries_to_features)


    def to_CNF(self, instance, target_prediction=None, *, tree_encoding=Encoding.COMPLEMENTARY, format=True, inverse_coding=False):
        """
        Two method:
        - TSEITIN: Create a DNF, i.e. a disjunction of cubes.
        Each cube is a model when the observation take the good prediction.
        Then the Tseitin transformation is applied in order to obtain a CNF.
        - COMPLEMENTARY: Create a DNF that is true when the observation take the wrong prediction.
        And we take the complementary of this DNF to obtain a CNF that is true when the observation takes the good prediction.
        """
        code_prediction = tree_encoding == Encoding.TSEITIN
        if inverse_coding:
            code_prediction = not code_prediction

        # Warning here, compute the prediction for a tree, not for a forest, this is not will exist !
        if target_prediction is None:
            target_prediction = self.predict_instance(instance)
        # Start to create the DNF according to the method TSEITIN or COMPLEMENTARY
        dnf = []
        for node in self.compute_nodes_with_leaves(self.root):
            if node.left.is_leaf() and \
                    ((code_prediction and node.left.is_prediction(target_prediction))
                     or (not code_prediction and not node.left.is_prediction(target_prediction))):
                dnf.append(self.create_cube(node, TypeLeaf.LEFT))

            if node.right.is_leaf() and \
                    ((code_prediction and node.right.is_prediction(target_prediction))
                     or (not code_prediction and not node.right.is_prediction(target_prediction))):
                dnf.append(self.create_cube(node, TypeLeaf.RIGHT))

        if tree_encoding == Encoding.COMPLEMENTARY:
            return CNFencoding.format(CNFencoding.complementary(dnf)) if format else CNFencoding.complementary(dnf)
        return CNFencoding.format(CNFencoding.tseitin(dnf)) if format else CNFencoding.tseitin(dnf)


    def create_cube(self, node, type_leaf):
        sign = -1 if type_leaf == TypeLeaf.LEFT else 1
        cube = [sign * self.get_id_variable(node)]
        parent = node.parent
        previous = node
        while parent is not None:
            sign = -1 if isinstance(parent.left, DecisionNode) and parent.left == previous else 1
            cube.append(sign * self.get_id_variable(parent))
            previous = parent
            parent = parent.parent
        return cube
