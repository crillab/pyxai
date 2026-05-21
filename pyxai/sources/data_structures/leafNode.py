
class LeafNode:
    def __init__(self, value):
        """
        A ```LeafNode``` represents the value of a leaf of a tree. 

        Parameters
        ----------
        value : ```int``` ```float```
            The value of the leaf.  

        Returns
        -------
        ```LeafNode```:
            The leaf node.

        Notes
        -----
        This method is usually not called directly, but through the ```DecisionNode``` class.
        """
        self.value = value
        self.parent = None

    def get_reachable_classes(self, reason, target_prediction, map_features_to_id_binaries):
        return set([self.value])

    def _is_implicant(self, reason, target_prediction, map_features_to_id_binaries, stack=[]):
        """_summary_

        Args:
            reason (_type_): The reason to check
            target_prediction (_type_): The actual target prediction
            map_features_to_id_binaries (_type_): data to get the variable ids from nodes

        Returns:
            Boolean: True if the reason is an implicant, else False
        """
        if self.value == target_prediction:
            return True
        return False


    def _take_decisions_binary_representation(self, binary_representation=None, map_features_to_id_binaries=None):
        return self.value


    def _take_decisions_instance(self, instance=None):
        return self.value


    def negating_tree(self):
        assert self.value == 1 or self.value == 0, "The negation of a tree is only possible with a 0 or 1 leaf value !"
        self.value = 1 if self.value == 0 else 0


    def is_prediction(self, prediction):
        return self.value == prediction


    def is_leaf(self):
        return True


    def __str__(self):
        return "leaf: {}".format(self.value)
