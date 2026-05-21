import os

from pyxai.sources.data_structures.boostedTrees import BoostedTrees
from pyxai.sources.learners.learner_information import LearnerInformation

class BoostedTreesRegression(BoostedTrees):
    def __init__(self, forest, learner_information=None, feature_names=None):
        """
        Returns a ```BoostedTreesRegression``` consisting of trees (the ```forest```).

        Parameters
        ----------
        forest : ```list``` of ```DecisionTree```
            The list of trees composing the boosted trees.  
        learner_information : ```LearnerInformation``` (optional, default=LearnerInformation(problem_type='regression', extras={"base_score": 0}) if called from Builder else None)
            The information about the learning process.  
        feature_names : ```list``` of ```str``` (optional, default=None)
            The names of the features used in the model.  
        class_names : ```list``` of ```str``` (optional, default=None)
            The names of the classes to predict.

        Returns
        -------
        ```BoostedTreesRegression```:
            The boosted regression trees model.
        
        Examples
        --------
        >>> from pyxai import Builder, Explaining

        >>> node1_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=3000, left=1500, right=1750)
        >>> node1_2 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2000, left=1000, right=node1_1)
        >>> node1_3 = Builder.DecisionNode(1, operator=Builder.GT, threshold=1000, left=0, right=node1_2)
        >>> tree1 = Builder.DecisionTree(5, node1_3)


        >>> node2_1 = Builder.DecisionNode(5, operator=Builder.EQ, threshold=1, left=100, right=250)
        >>> node2_2 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-100, right=node2_1)
        >>> node2_3 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=node2_2, right=250)
        >>> tree2 = Builder.DecisionTree(5, node2_3)

        >>> node3_1 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=500, right=250)
        >>> node3_2 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=250, right=100)
        >>> node3_3 = Builder.DecisionNode(1, operator=Builder.GE, threshold=2000, left=0, right=node3_1)
        >>> node3_4 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=node3_3, right=node3_2)
        >>> tree3 = Builder.DecisionTree(5, node3_4)


        >>> BTs = Builder.BoostedTreesRegression([tree1, tree2, tree3])
        """
        super().__init__(forest, None, learner_information=learner_information)

    def __str__(self):
        s = "**Boosted Tree Regression model**" + os.linesep
        s += "nTrees: " + str(self.n_trees) + os.linesep
        # s += "nFeatures in the biggest tree: " + str(max(tree.existing_variables() for tree in self.forest)) + os.linesep
        s += "nVariables: " + str(len(self.map_id_binaries_to_features) - 1) + os.linesep
        return s

    def _predict_implicant(self, implicant):
        """
        Return the prediction of an implicant according to the trees
        """
        base_score = self.learner_information.extras["base_score"]
        sum_trees = sum([tree._take_decisions_binary_representation(implicant, self.map_features_to_id_binaries) for tree in self.forest])
        return sum_trees + base_score 
    
    def _predict_instance(self, instance):
        """
        Return the prediction of an instance according to the trees
        """
        base_score = self.learner_information.extras["base_score"]
        sum_trees = sum([tree._predict_instance(instance) for tree in self.forest])
        return sum_trees + base_score 
