from pyxai.sources.data_structures.binaryMapping import BinaryMapping
from pyxai.sources.data_structures.decisionTree import DecisionTree

class TreeEnsembles(BinaryMapping):
    """
    Represent a set of trees. This class is used for the class RandomForest (RF) and BoostedTrees (BTs)
    map_id_binaries_to_features: list[id_binary] -> (id_feature, op, threshold)
    map_features_to_id_binaries: dict[(id_feature, operator, threshold)] -> [id_binary, n_appears, n_appears_per_tree]
    """


    def __init__(self, forest):
        self.forest = forest
        self.n_trees = len(forest)
        self.force_features_equal_to_binaries = forest[0].force_features_equal_to_binaries
              
        assert all(isinstance(tree, DecisionTree) for tree in forest), "All trees in the forest have to be of the type DecisionTree."
        assert all(tree.force_features_equal_to_binaries == self.force_features_equal_to_binaries for tree in
                   forest), "All trees in the forest have to have the same force_features_equal_to_binaries value."

        if self.force_features_equal_to_binaries is False:
            assert all(tree.n_features == self.forest[0].n_features for tree in forest), "All trees in the forest have to have the same number of input (features)."
        
        self.map_id_binaries_to_features, self.map_features_to_id_binaries = self._compute_id_binaries(self.force_features_equal_to_binaries)
        super().__init__(self.map_id_binaries_to_features, self.map_features_to_id_binaries)

        # Change the encoding of each tree by these new encoding
        for tree in self.forest:
            tree.map_id_binaries_to_features = self.map_id_binaries_to_features
            tree.map_features_to_id_binaries = self.map_features_to_id_binaries

    def n_nodes(self):
        return sum(tree.n_nodes() for tree in self.forest)
    
    def depth(self):
        return max(tree.depth() for tree in self.forest)
    

    def _compute_id_binaries(self, force_features_equal_to_binaries=False):
        """
        Overload method from the mother class BinaryMapping
        map_id_binaries_to_features: list[id_binary] -> (id_feature, operator, threshold)
        map_features_to_id_binaries: dict[(id_feature, operator, threshold)] -> [id_binary, n_appears, n_appears_per_tree]
        """

        if not force_features_equal_to_binaries:
            map_id_binaries_to_features = [0]
        else:
            map_id_binaries_to_features = [0] + [None] * self.forest[0].n_features

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
