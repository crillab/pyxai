import math
import numpy
import os
from operator import index

from pyxai.sources.core.structure.decisionTree import DecisionTree
from pyxai.sources.core.structure.treeEnsembles import TreeEnsembles
from pyxai.sources.core.structure.type import Encoding
from pyxai.sources.core.tools.encoding import CNFencoding


class BoostedTrees(TreeEnsembles):

    def __init__(self, forest, n_classes=2, learner_information=None):
        super().__init__(forest, learner_information)
        self.n_classes = n_classes
        self.learner_information = learner_information
        # assert all(tree.type_tree is TypeTree.WEIGHT for tree in self.forest), "All trees in a boosted trees have to be of the type WEIGHT."


    def raw_data(self):
        raw = tuple(tree.raw_data() for tree in self.forest)
        return (self.n_classes, raw)


    def get_set_of_variables(self, tree, node):
        output = set()
        if node.is_leaf(): return output
        output.add(tree.get_id_variable(node))
        output = output.union(self.get_set_of_variables(tree, node.right))
        output = output.union(self.get_set_of_variables(tree, node.left))
        return output


    def get_leaves(self, tree, node, implicant, is_removed, current_removed=None, is_in_current=False):
        output = []
        if node.is_leaf():
            if current_removed is not None:
                output.append((node, is_in_current))
            else:
                output.append(node)
        else:
            id_variable = tree.get_id_variable(node)
            index_variable = implicant.index(id_variable) if id_variable in implicant else implicant.index(-id_variable)
            literal = implicant[index_variable]
            removed = is_removed[index_variable]
            if current_removed is not None:
                if index_variable == current_removed:
                    is_in_current = True
            if removed:
                output += self.get_leaves(tree, node.right, implicant, is_removed, current_removed, is_in_current)
                output += self.get_leaves(tree, node.left, implicant, is_removed, current_removed, is_in_current)
            else:
                if literal >= 0:
                    output += self.get_leaves(tree, node.right, implicant, is_removed, current_removed, is_in_current)
                else:
                    output += self.get_leaves(tree, node.left, implicant, is_removed, current_removed, is_in_current)
        return output


    def reduce_nodes(self, node, tree, implicant, get_min):
        if node.is_leaf():
            return
        self.reduce_nodes(node.left, tree, implicant, get_min)
        self.reduce_nodes(node.right, tree, implicant, get_min)
        if node.left.is_leaf() and node.right.is_leaf():
            id_variable = tree.get_id_variable(node)
            instance_w = node.right.value if id_variable in implicant else node.left.value
            not_instance_w = node.left.value if id_variable in implicant else node.right.value
            if (get_min and instance_w < not_instance_w) or (not get_min and instance_w > not_instance_w):
                node.artificial_leaf = True
                node.value = instance_w


    def reduce_trees(self, implicant, prediction):
        for tree in self.forest:
            for node in tree.nodes:
                node.artificial_leaf = False
            self.reduce_nodes(tree.root, tree, implicant, prediction == 1 if self.n_classes == 2 else tree.target_class == prediction)


    def remove_reduce_trees(self):
        for tree in self.forest:
            for node in tree.nodes:
                node.artificial_leaf = False


    def __str__(self):
        s = "**Boosted Tree model**" + os.linesep
        s += "NClasses: " + str(self.n_classes) + os.linesep
        s += "nTrees: " + str(self.n_trees) + os.linesep
        # s += "nFeatures in the biggest tree: " + str(max(tree.existing_variables() for tree in self.forest)) + os.linesep
        s += "nVariables: " + str(len(self.map_id_binaries_to_features) - 1) + os.linesep
        return s


    def scores_to_probabilities(self, scores):
        if self.n_classes > 2:
            class_scores = numpy.asarray([(math.exp((scores[i::self.n_classes]).sum())) for i in range(self.n_classes)])
            return class_scores / class_scores.sum()
        else:
            # Remark: it is not probabilities here but it is ok
            return [0, 1] if sum(scores) > 0 else [1, 0]


    def compute_probabilities_instance(self, instance):
        scores = numpy.asarray([tree.predict_instance(instance) for tree in self.forest])
        return self.scores_to_probabilities(scores)


    def compute_probabilities_implicant(self, implicant):
        scores = numpy.asarray([tree.take_decisions_binary_representation(implicant, self.map_features_to_id_binaries) for tree in self.forest])
        return self.scores_to_probabilities(scores)


    def predict_implicant(self, implicant):
        """
        Return the prediction (the classification) of an implicant according to the trees
        """
        return numpy.argmax(self.compute_probabilities_implicant(implicant))


    def predict_instance(self, instance):
        """
        Return the prediction (the classification) of an instance according to the trees
        """
        return numpy.argmax(self.compute_probabilities_instance(instance))
