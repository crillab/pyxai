import math
import numpy
import os
from operator import index
from decimal import Decimal
from pyxai.sources.data_structures.decisionTree import DecisionTree
from pyxai.sources.data_structures.treeEnsembles import TreeEnsembles
from pyxai.sources.type import Encoding
from pyxai.sources.tools.encoding import CNFencoding
from pyxai.sources.learners.learner_information import LearnerInformation
from pyxai.sources.data_structures.binaryMapping import BinaryMapping
from pyxai.sources.data_structures.baseModel import BaseModel
import xgboost
import pandas

# TODO fix circular import
class BoostedTrees(TreeEnsembles, BaseModel):

    def __init__(self, forest, n_classes=None, learner_information=None, feature_names=None, class_names=None):
        """
        Returns a ```BoostedTrees``` consisting of trees (the ```forest```) and ```n_classes``` to predict.

        Parameters
        ----------
        forest : ```list``` of ```DecisionTree```
            The list of trees composing the boosted trees.  
        n_classes : ```int```
            The number of classes to predict (can be multi-class).  
        learner_information : ```LearnerInformation``` (optional, default=LearnerInformation(problem_type='classification', extras={"base_score": 0}) if called from Builder else None)
            The information about the learning process.  
        feature_names : ```list``` of ```str``` (optional, default=None)
            The names of the features used in the model.  
        class_names : ```list``` of ```str``` (optional, default=None)
            The names of the classes to predict.
        
        Returns
        -------
        ```BoostedTrees```:
            The boosted trees model.
        
        Examples
        --------
        >>> from pyxai import Builder

        >>> node1_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=-0.2, right=0.3)
        >>> node1_2 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=-0.3, right=node1_1)
        >>> node1_3 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=0.4, right=node1_2)
        >>> node1_4 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.5, right=node1_3)
        >>> tree1 = Builder.DecisionTree(4, node1_4)

        >>> node2_1 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.4, right=0.3)
        >>> node2_2 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=-0.2, right=node2_1)
        >>> node2_3 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=node2_2, right=0.5)
        >>> tree2 = Builder.DecisionTree(4, node2_3)

        >>> node3_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=0.2, right=0.3)

        >>> node3_2_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=-0.2, right=0.2)

        >>> node3_2_2 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.1, right=node3_1)
        >>> node3_2_3 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.5, right=0.1)

        >>> node3_3_1 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=node3_2_1, right=node3_2_2)
        >>> node3_3_2 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=-0.4, right=node3_2_3)
        >>> node3_4 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=node3_3_1, right=node3_3_2)

        >>> tree3 = Builder.DecisionTree(4, node3_4)

        >>> BTs = Builder.BoostedTrees([tree1, tree2, tree3], n_classes=2)

        """

        BaseModel.__init__(self, None, n_classes, learner_information, feature_names, class_names)
        TreeEnsembles.__init__(self, forest)
        
        
        # assert all(tree.type_tree is TypeTree.WEIGHT for tree in self.forest), "All trees in a boosted trees have to be of the type WEIGHT."  Not used 19/09/2025


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


    def get_leaves(self, tree, node, implicant, is_removed):
        output = []
        if node.is_leaf():
            output.append(node)
        else:
            id_variable = tree.get_id_variable(node)
            print("id_variable:", id_variable)
            print("implicant:", implicant)
            index_variable = implicant.index(id_variable) if id_variable in implicant else implicant.index(-id_variable)
            literal = implicant[index_variable]
            removed = is_removed[index_variable]
            if removed:
                output += self.get_leaves(tree, node.right, implicant, is_removed)
                output += self.get_leaves(tree, node.left, implicant, is_removed)
            else:
                if literal >= 0:
                    output += self.get_leaves(tree, node.right, implicant, is_removed)
                else:
                    output += self.get_leaves(tree, node.left, implicant, is_removed)
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
            for node in tree._nodes:
                node.artificial_leaf = False
            self.reduce_nodes(tree._root, tree, implicant, prediction == 1 if self.n_classes == 2 else tree._associated_class == prediction)


    def remove_reduce_trees(self):
        for tree in self.forest:
            for node in tree._nodes:
                node.artificial_leaf = False


    def __str__(self):
        s = "**Boosted Tree model**" + os.linesep
        s += "NClasses: " + str(self.n_classes) + os.linesep
        s += "nTrees: " + str(self.n_trees) + os.linesep
        # s += "nFeatures in the biggest tree: " + str(max(tree.existing_variables() for tree in self.forest)) + os.linesep
        s += "nVariables: " + str(len(self.map_id_binaries_to_features) - 1) + os.linesep
        return s


    def _scores_to_probabilities(self, scores):
        # -numpy.log(1/base_score-1) present in XGBoost code
        # See in src/objective/regression_loss.h of XGBoost
        # For the moment, it is not implemented for LightGBM
        
        if self.n_classes > 2:
            # In the multi-class case, we do not need to add the base_score because increasing the scores of all classes by an equal amount changes nothing. 
            class_scores = numpy.asarray([((scores[i::self.n_classes]).sum()) for i in range(self.n_classes)])
            class_scores += self.learner_information.extras["base_score"]
            inter_softmax_class_scores = numpy.exp(class_scores - numpy.max(class_scores)) #Intermediate step for computing softmax
            return inter_softmax_class_scores / inter_softmax_class_scores.sum(axis=0)
        
        else:
            # Remark: it is not probabilities here but it is ok
            # In the binary case, we need to add the base_score
            base_score = self.learner_information.extras["base_score"] 
            base_score = -numpy.log(1 / base_score - 1) if base_score != 0 else 0
            return [0, 1] if scores.sum() + base_score > 0 else [1, 0]

            

    def _predict_implicant(self, implicant):
        """
        Return the prediction (the classification) of an implicant according to the trees
        """
        scores = numpy.asarray([tree._take_decisions_binary_representation(implicant, self.map_features_to_id_binaries) for tree in self.forest])
        return numpy.argmax(self._scores_to_probabilities(scores))

    def _predict_instance(self, instance):
        """
        Return the prediction (the classification) of an instance according to the trees
        """
        scores = numpy.asarray([tree._predict_instance(instance) for tree in self.forest])
        return numpy.argmax(self._scores_to_probabilities(scores))

