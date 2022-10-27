from typing import Iterable

from pyxai.sources.core.structure.binaryMapping import BinaryMapping
from pyxai.sources.core.structure.decisionTree import DecisionTree


class TreeEnsembles(BinaryMapping):
    """
    Represent a set of trees. This class is used for the class RandomForest (RF) and BoostedTrees (BTs)
    map_id_binaries_to_features: list[id_binary] -> (id_feature, threshold)
    map_features_to_id_binaries: dict[(id_feature, threshold)] -> [id_binary, n_appears, n_appears_per_tree]
    """


    def __init__(self, forest, learner_information):
        self.forest = forest
        self.n_trees = len(forest)
        self.n_features = forest[0].n_features
        self.force_features_equal_to_binaries = forest[0].force_features_equal_to_binaries

        # TO DO: multi-classes in RF ???
        self.classes = set()
        for tree in self.forest:
            if isinstance(tree.target_class, Iterable):
                self.classes = self.classes.union(set(list(tree.target_class)))
            else:
                self.classes.add(tree.target_class)

        assert all(isinstance(tree, DecisionTree) for tree in forest), "All trees in the forest have to be of the type DecisionTree."
        assert all(tree.n_features == self.n_features for tree in forest), "All trees in the forest have to have the same number of input (features)."
        assert all(tree.force_features_equal_to_binaries == self.force_features_equal_to_binaries for tree in
                   forest), "All trees in the forest have to have the same force_features_equal_to_binaries value."

        self.map_id_binaries_to_features, self.map_features_to_id_binaries = self.compute_id_binaries(self.force_features_equal_to_binaries)
        super().__init__(self.map_id_binaries_to_features, self.map_features_to_id_binaries, learner_information)

        # Change the encoding of each tree by these new encoding
        for tree in self.forest:
            tree.map_id_binaries_to_features = self.map_id_binaries_to_features
            tree.map_features_to_id_binaries = self.map_features_to_id_binaries


    def redundancy_analysis(self):
        n_variables = len(self.map_features_to_id_binaries)
        n_alone_variables = 0
        n_redundant_variables = 0
        redundancy = []

        for key in self.map_features_to_id_binaries.keys():
            n_appears_in_the_same_tree = max(self.map_features_to_id_binaries[key][2])
            n_appears_in_distinct_tree = sum(1 if value > 0 else 0 for value in self.map_features_to_id_binaries[key][2])
            print("for key:", key)
            print("n_appears:", self.map_features_to_id_binaries[key][1])
            print("n_appears_in_the_same_tree:", n_appears_in_the_same_tree)
            print("n_appears_in_distinct_tree:", n_appears_in_distinct_tree)
            if n_appears_in_distinct_tree == 1:
                n_alone_variables += 1
            else:
                n_redundant_variables += 1
                redundancy.append(n_appears_in_distinct_tree)
        print("n_variables: ", n_variables)
        print("n_alone_variables: ", n_alone_variables)
        print("n_redundant_variables: ", n_redundant_variables)
        print("n_redundant_variables: ", n_redundant_variables)
        print("redundancy:", redundancy)
        print("avg redundancy:", sum(redundancy) / len(redundancy))


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

        # Fusion of map_id_binaries_to_features
        for tree in self.forest:
            map_features_to_id_binaries.update(tree.map_features_to_id_binaries)

        # Now we define the good value [id_binary, n_appears, n_appears_per_tree] for each key
        for key in map_features_to_id_binaries.keys():  # the keys are of the type: (id_feature, operator, threshold)
            if not force_features_equal_to_binaries:  # Do not touch this when force_features_equal_to_binaries is True because id_binary do not move
                map_features_to_id_binaries[key][0] = id_binary

            values = [tree.map_features_to_id_binaries.get(key) for tree in self.forest]
            n_appears = sum(value[1] for value in values if value is not None)
            n_appears_per_tree = [value[1] if value is not None else 0 for value in values]
            map_features_to_id_binaries[key][1] = n_appears
            map_features_to_id_binaries[key][2] = n_appears_per_tree

            if force_features_equal_to_binaries is False:
                map_id_binaries_to_features.append(key)
            else:
                map_id_binaries_to_features[key[0]] = key

            id_binary += 1

        return (map_id_binaries_to_features, map_features_to_id_binaries)
