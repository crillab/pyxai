import copy
import numpy
import os

from pyxai.sources.learners.learner_information import LearnerInformation
from pyxai.sources.data_structures.binaryMapping import BinaryMapping
from pyxai.sources.data_structures.decisionNode import DecisionNode, LeafNode
from pyxai.sources.type import TypeLeaf, Encoding, OperatorCondition
from pyxai.sources.tools.encoding import CNFencoding
from pyxai.sources.data_structures.baseModel import BaseModel

class DecisionTree(BinaryMapping, BaseModel):

    def __init__(self, n_features, root, n_classes=None, force_features_equal_to_binaries=False, learner_information=None, feature_names=None, class_names=None):
        r"""
        Returns a ```DecisionTree``` of ```n_features``` which has as root node the ```root``` parameter. 
        
        Parameters
        ----------
        n_features : ```int```
            The total number of features used by all trees of the model (not only those of this ```DecisionTree```)
        root : ```DecisionNode``` or ```LeafNode```
            The root node of the decision tree.
        force_features_equal_to_binaries : ```bool```, default=```False```
            Setting this parameter to ```True``` ensures that the binary variables are equal to feature identifiers. 
            Assume that a feature represents a condition (this is not true with most datasets). 
            Put this option to ```True``` only when you build a tree where all features are Boolean conditions.
        learner_information : ```LearnerInformation``` (optional, default=LearnerInformation(problem_type='classification') if called from Builder else None)
            The learner information associated to this tree.
        feature_names : ```list``` of ```str``` (optional, default=None)
            The names of the features used in this tree.
        class_names : ```list``` of ```str``` (optional, default=None)
            The names of the classes used in this tree.
        
        Returns
        -------
        ```DecisionTree```:
            The decision tree model.
        
        Example
        -------
        >>> node_v3_1 = Builder.DecisionNode(3, operator="EQ", threshold=1, left=0, right=1)
        >>> node_v2_1 = Builder.DecisionNode(2, operator="EQ", threshold=1, left=0, right=node_v3_1)
        >>> node_v3_2 = Builder.DecisionNode(3, operator="EQ", threshold=1, left=0, right=1)
        >>> node_v2_2 = Builder.DecisionNode(2, operator="EQ", threshold=1, left=0, right=node_v3_2)
        >>> node_v3_3 = Builder.DecisionNode(3, operator="EQ", threshold=1, left=0, right=1)
        >>> node_v2_3 = Builder.DecisionNode(2, operator="EQ", threshold=1, left=0, right=node_v3_3)
        >>> node_v1_1 = Builder.DecisionNode(1, operator="GE", threshold=10, left=node_v2_1, right=node_v2_2)
        >>> node_v1_2 = Builder.DecisionNode(1, operator="GE", threshold=20, left=node_v1_1, right=node_v2_3)
        >>> node_v1_3 = Builder.DecisionNode(1, operator="GE", threshold=30, left=node_v1_2, right=1)
        >>> node_v1_4 = Builder.DecisionNode(1, operator="GE", threshold=40, left=node_v1_3, right=1)
        >>> tree = Builder.DecisionTree(3, node_v1_4)
        
        Notes   
        -----
        When the root parameter is a ```LeafNode```, the tree contains only one node (the root).
        """

        # The root node of the tree
        self._root = root

        # All nodes of the tree that are not leaves
        self._nodes = []

        # Variable used to save the leaves of the tree
        self._leaves = None
        
        # Classes that are associated to this tree according to the model used:
        # For classification: DT and RF: A list of all classes because all classes are used for each tree
        # For classification: BT: None for binary classification. An integer representing the class for each tree in the multiclass case.  
        # For regression: not used => None
        
        # Only in the case of multiclass BT, it is an integer representing the class for each tree, else None  
        self._associated_class = None 
        
        # By default, the binaries id in the representation of implicants
        # is not the same that the features id. This option allows to force that these two kinds of ids are the same. This is used to allows
        # that variables directly represent the features. Defaults to False.
        self.force_features_equal_to_binaries = force_features_equal_to_binaries

        if not self._root.is_leaf():
            self.define_parents(self._root)

        BaseModel.__init__(self, n_features, n_classes, learner_information, feature_names, class_names)

        map_id_binaries_to_features, map_features_to_id_binaries = self._compute_id_binaries(self.force_features_equal_to_binaries)
        BinaryMapping.__init__(self, map_id_binaries_to_features, map_features_to_id_binaries)


    def set_associated_class(self, _associated_class):
        self._associated_class = _associated_class
        

    def __str__(self):
        s = "**Decision Tree Model**" + os.linesep
        s += "nFeatures: " + str(self.n_features) + os.linesep
        s += "nNodes: " + str(len(self._nodes)) + os.linesep
        s += "nVariables: " + str(len(self.map_id_binaries_to_features) - 1) + os.linesep
        return s

    def from_tuples(self, tuples):
        if isinstance(tuples, int):
            return LeafNode(tuples)
 
        binary_variable = tuples[0]
        id_feature, op, threshold = self.map_id_binaries_to_features[binary_variable]
        
        node = DecisionNode(id_feature, threshold=threshold, operator=op, left=None, right=None)
        node.left = self.from_tuples(tuples[1][0])
        node.right = self.from_tuples(tuples[1][1])
        return node

    
    def delete(self, node):
        if node.is_leaf(): 
            del node
        else:
            self.delete(node.left)
            self.delete(node.right)
            del node

    def raw_data_for_CPP(self):
        raw_t = tuple([self._root.value]) if self._root.is_leaf() else self.to_tuples(self._root, for_cpp=True)
        return (0 if self._associated_class is None else self._associated_class, raw_t) 


    def raw_data(self):
        raw = tuple([self._root.value]) if self._root.is_leaf() else self.to_tuples(self._root)
        if type(self._associated_class) is int:
            return (int(self.n_features), [self._associated_class], raw)
        else:
            return (int(self.n_features), [int(element) for element in self._associated_class], raw)

    def depth(self):
        return self._depth(self._root) 
    

    def _depth(self, node):
        if node.is_leaf():
            return 1  
        left_depth = self._depth(node.left)
        right_depth = self._depth(node.right)
        return max(left_depth, right_depth) + 1
        
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
        while self._simplify(self._root, self._root):
            pass
        raw = self.to_tuples(self._root)
        if raw[1] == raw[2]:
            self._root = self._root.left
        


    def _simplify(self, root, node, path=[], come_from=None, previous_node=None, previous_previous_node=None):
        res_1 = False
        res_2 = False
        change = False

        
        if previous_node is not None:
            new_tuple = (self.get_id_variable(previous_node), come_from)
            if new_tuple in path:
                if path[-1][1] == 0:
                    if previous_previous_node is not None:
                        previous_previous_node.left = node
                        change = True
                elif path[-1][1] == 1:
                    if previous_previous_node is not None:
                        previous_previous_node.right = node
                        change = True
            path.append(new_tuple)

        if not node.is_leaf():
            raw = self.to_tuples(node)
            if raw[1] == raw[2]:
                if come_from == 0:
                    if previous_node is not None:
                        previous_node.left = node.left
                        change = True
                if come_from == 1:
                    if previous_node is not None:
                        previous_node.right = node.right
                        change = True
            pp = previous_node
            res_1 = self._simplify(root, node.left, copy.deepcopy(path), come_from=0, previous_node=node, previous_previous_node=pp)
            res_2 = self._simplify(root, node.right, copy.deepcopy(path), come_from=1, previous_node=node, previous_previous_node=pp)
        return res_1 or res_2 or change

    """
        Transform a decision rule into a Decision Tree (DT).
        Args:
            decision_rule (list or tuple): A decision rule in the form of list of literals (binary variables representing the conditions of the tree). 
        Returns:
            DecisionTree: A decision tree representing the decision rule.  
    """
    def decision_rule_to_tree(self, decision_rule, label):
        if len(decision_rule) == 0:
            tree = DecisionTree(self.n_features, LeafNode(label))
            tree.map_id_binaries_to_features = self.map_id_binaries_to_features
            tree.map_features_to_id_binaries = self.map_features_to_id_binaries
            return tree

        literal = decision_rule[-1]
        
        id_feature, operator, threshold  = self.map_id_binaries_to_features[abs(literal)]
        parent = DecisionNode(id_feature, operator=operator, threshold=threshold, left=1, right=0) if literal > 0 else DecisionNode(id_feature, operator=operator, threshold=threshold, left=0, right=1)
        
        for literal in reversed(decision_rule[:-1]):
            id_feature, operator, threshold = self.map_id_binaries_to_features[abs(literal)]
            parent = DecisionNode(id_feature, operator=operator, threshold=threshold, left=1, right=parent) if literal > 0 else DecisionNode(id_feature,operator=operator, threshold=threshold, left=parent, right=1)
        
        tree = DecisionTree(self.n_features, parent)

        #This tree have to have the same data of the initial tree !
        tree.map_id_binaries_to_features = self.map_id_binaries_to_features
        tree.map_features_to_id_binaries = self.map_features_to_id_binaries
        
        return tree

    


    def negating_tree(self):
        new_tree = copy.deepcopy(self)
        new_tree._root.negating_tree()
        return new_tree


    def concatenate_tree(self, other_tree):
        new_tree = copy.deepcopy(self)
        new_tree._root.concatenate_tree(other_tree)
        new_tree._concatenate_id_binaries(other_tree)
        return new_tree


    def disjoint_tree(self, other_tree):
        new_tree = copy.deepcopy(self)
        new_tree._root.concatenate_tree(other_tree, disjunction=True)
        new_tree._concatenate_id_binaries(other_tree)
        return new_tree


    def get_variables(self, binary_representation=None, node=None):
        if node is None:
            if self._root.is_leaf():
                return []
            node = self._root
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
            if self._root.is_leaf():
                return []
            node = self._root

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
            node = self._root
        if node.is_leaf():
            return []
        output = []
        value = numpy.array(instance)[node.id_feature - 1]
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
            self._nodes.append(node)
            self.define_parents(node.left, parent=node)
            self.define_parents(node.right, parent=node)
        if parent is not None:
            node.parent = parent


    def _concatenate_id_binaries(self, other_tree):
        self.map_features_to_id_binaries.update(other_tree.map_features_to_id_binaries)
        for i, element in enumerate(self.map_id_binaries_to_features):
            if element is None:
                self.map_id_binaries_to_features[i] = other_tree.map_id_binaries_to_features[i]


    def _compute_id_binaries(self, force_features_equal_to_binaries=False):
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
        for node in self._nodes:
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
        return self._root.is_leaf()


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

    def n_nodes(self):
        return self._n_nodes(self._root)

    def _n_nodes(self, node):
        return 1 if node.is_leaf() else 1 + self._n_nodes(node.left) + self._n_nodes(node.right)
        
    def display(self, node):
        if node.is_leaf():
            print(node)
        else:
            print(node)
            self.display(node.left)
            self.display(node.right)

    ###
    # Do not use directly, as the implicant cannot coincide with the theory 
    # (use is_implicant of explainer object instead)
    ###
    def _is_implicant(self, reason, target_prediction):
        if self.class_names is not None and type(target_prediction) == str:
            target_prediction = numpy.where(self.class_names==target_prediction)[0][0]
        return self._root._is_implicant(reason, target_prediction, self.map_features_to_id_binaries)

    
    def get_reachable_classes(self, reason, target_prediction):
        return self._root.get_reachable_classes(reason, target_prediction, self.map_features_to_id_binaries)

    
    def _predict_instance(self, instance):
        """
        Return the prediction (the classification) of an observation (instance) according to this tree
        """
        return self._root._take_decisions_instance(instance)

        
    def _predict_implicant(self, implicant, map_features_to_id_binaries=None):
        return self._take_decisions_binary_representation(implicant, map_features_to_id_binaries)

    
    def _take_decisions_binary_representation(self, binary_representation, map_features_to_id_binaries=None):
        """
        Return the prediction (the classification) of an binary representation according to this tree
        """
        if map_features_to_id_binaries is None:
            map_features_to_id_binaries = self.map_features_to_id_binaries
        return self._root._take_decisions_binary_representation(binary_representation, map_features_to_id_binaries)


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
            target_prediction = self._predict_instance(instance)
        # Start to create the DNF according to the method TSEITIN or COMPLEMENTARY
        dnf = []
        if self._root.is_leaf():
            return []
            
        for node in self.compute_nodes_with_leaves(self._root):
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


    def _get_leaves(self, node) :
        if node.is_leaf() :
            return [node]
        return self._get_leaves(node.left) + self._get_leaves(node.right)

    def get_leaves(self):
        if self._leaves is None:
            self._leaves = self._get_leaves(self._root)
        return self._leaves

    def get_min_value(self):
        return min([l.value for l in self.get_leaves()])

    def get_max_value(self):
        return max([l.value for l in self.get_leaves()])