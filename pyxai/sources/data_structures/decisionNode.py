import copy
import pandas

from pyxai.sources.type import OperatorCondition
from pyxai.sources.data_structures.leafNode import LeafNode

class DecisionNode:
    """
    A decision node represent a decision. A decision tree consists of these nodes.
    """


    def __init__(self, id_feature, *, threshold=0.5, operator=OperatorCondition.GE, left, right, parent=None):
        r"""
        A ```DecisionNode``` represents a condition "\<id_feature\> \<operator\> \<threshold\> ?" (such as "$x_4 ≥ 0.5$ ?") in the model while a ```LeafNode``` is a value.
        During the process, can also create two or one ```LeafNode``` without returning it.

        Parameters
        ----------
        id_feature : ```int```
            The feature identifier used in the condition "\<id_feature\> \<operator\> \<threshold\> ?".

        operator : ```OperatorCondition``` ```str```, default=```OperatorCondition.GE```
            The operator used in the condition "\<id_feature\> \<operator\> \<threshold\> ?".
            Possible values are defined in the ```OperatorCondition``` enum.

        threshold : ```float```, default=```0.5```
            The threshold used in the condition "\<id_feature\> \<operator\> \<threshold\> ?".
            
        parent : ```DecisionNode``` ```None``` (optional, default=None)
            To define the parent of this node. If this parameter is set to ```None```, the parent is automatically defined when the tree is created.
        
        left : ```DecisionNode``` ```Integer``` ```Float```
            The left child of the node. When this parameter is an ```Integer``` or a ```Float```, a ```LeafNode``` is generated.

        right: ```DecisionNode``` ```Integer``` ```Float```
            The left child of the node. When this parameter is an ```Integer``` or a ```Float```, a ```LeafNode``` is generated.
        
        Returns
        -------
        ```DecisionNode```:
            The decision node.

        Examples
        --------
        >>> node_v3_1 = Builder.DecisionNode(3, operator="EQ", threshold=1, left=0, right=1)
        >>> node_v2_1 = Builder.DecisionNode(2, operator="EQ", threshold=1, left=0, right=node_v3_1)  

        Notes
        -----
        When the operator and threshold parameters are not defined, they take their default values. 
        In this case, the associated condition is of the form "$x ≥ 0.5$ ?"
        """
        self.id_feature = id_feature
        self.threshold = threshold
        self.operator = operator if isinstance(operator, OperatorCondition) else OperatorCondition(operator)
        self.parent = parent
        self.left = left if isinstance(left, DecisionNode) else LeafNode(left)
        self.right = right if isinstance(right, DecisionNode) else LeafNode(right)
        self.artificial_leaf = False

    
    def negating_tree(self):
        self.right.negating_tree()
        self.left.negating_tree()


    def concatenate_tree(self, other_tree, disjunction=False):
        if self.right.is_leaf():
            assert self.right.value == 1 or self.right.value == 0, "The concatenation of a tree is only possible with a 0 or 1 leaf value !"
            if not disjunction:
                if self.right.value == 1:
                    self.right = copy.deepcopy(other_tree._root)
            else:
                if self.right.value == 0:
                    self.right = copy.deepcopy(other_tree._root)
        else:
            self.right.concatenate_tree(other_tree, disjunction)

        if self.left.is_leaf():
            assert self.left.value == 1 or self.left.value == 0, "The concatenation of a tree is only possible with a 0 or 1 leaf value !"
            if not disjunction:
                if self.left.value == 1:
                    self.left = copy.deepcopy(other_tree._root)
            else:
                if self.left.value == 0:
                    self.left = copy.deepcopy(other_tree._root)

        else:
            self.left.concatenate_tree(other_tree, disjunction)


    def is_leaf(self):
        return self.artificial_leaf


    def __str__(self):
        return "f{} {} {}".format(self.id_feature, self.operator, self.threshold)


    def get_reachable_classes(self, binary_representation, target_prediction, map_features_to_id_binaries):
        id_variable = map_features_to_id_binaries[(self.id_feature, self.operator, self.threshold)][0]
        if id_variable in binary_representation:
            return self.right.get_reachable_classes(binary_representation, target_prediction, map_features_to_id_binaries)
        elif -id_variable in binary_representation:
            return self.left.get_reachable_classes(binary_representation, target_prediction, map_features_to_id_binaries)
        else:
            right = self.right.get_reachable_classes(binary_representation, target_prediction, map_features_to_id_binaries)
            left = self.left.get_reachable_classes(binary_representation, target_prediction, map_features_to_id_binaries)
            return right.union(left)

    def _is_implicant(self, binary_representation, target_prediction, map_features_to_id_binaries, stack=[]):
        """_summary_

        Args:
            binary_representation (_type_): The reason to check
            target_prediction (_type_): The actual target prediction
            map_features_to_id_binaries (_type_): data to get the variable ids from nodes

        Returns:
            Boolean: True if the reason is an implicant, else False
        """        
        id_variable = map_features_to_id_binaries[(self.id_feature, self.operator, self.threshold)][0]
        if id_variable in binary_representation:
            return self.right._is_implicant(binary_representation, target_prediction, map_features_to_id_binaries, stack=stack+[id_variable])
        elif -id_variable in binary_representation:
            return self.left._is_implicant(binary_representation, target_prediction, map_features_to_id_binaries, stack=stack+[-id_variable])
        else:
            right = self.right._is_implicant(binary_representation, target_prediction, map_features_to_id_binaries, stack=stack+[id_variable])
            left = self.left._is_implicant(binary_representation, target_prediction, map_features_to_id_binaries, stack=stack+[-id_variable])
            return right and left


    def _take_decisions_binary_representation(self, binary_representation, map_features_to_id_binaries):
        """
        Return the prediction (the classification) of a binary representation according to this node.
        Warning: The binary representation (propositional formula calculated from a instance) have to be complete
         (i.e only a direct reason, not a reason)
        Warning: right nodes are considered as the 'yes' responses of conditions, left nodes as 'no'.
        """
        id_variable = map_features_to_id_binaries[(self.id_feature, self.operator, self.threshold)][0]
        assert (id_variable in binary_representation) or (-id_variable in binary_representation), "The binary representation has to be complete !"
        if id_variable in binary_representation:
            return self.right._take_decisions_binary_representation(binary_representation, map_features_to_id_binaries)
        else:
            return self.left._take_decisions_binary_representation(binary_representation, map_features_to_id_binaries)


    def _take_decisions_instance(self, instance):
        """
        Return the prediction (the classification) of an observation (instance) according to this node.
        This return value is either 0 or 1: 0 for the first (boolean) prediction value, 1 for the second one.
        Warning: right nodes are considered as the 'yes' responses of conditions, left nodes as 'no'.
        """
        instance = pandas.Series(instance)        
        if self.operator == OperatorCondition.GE:
            if instance.iloc[self.id_feature - 1] >= self.threshold:
                return self.right._take_decisions_instance(instance)
            else:
                return self.left._take_decisions_instance(instance)
        elif self.operator == OperatorCondition.GT:
            if instance.iloc[self.id_feature - 1] > self.threshold:
                return self.right._take_decisions_instance(instance)
            else:
                return self.left._take_decisions_instance(instance)
        elif self.operator == OperatorCondition.LE:
            if instance.iloc[self.id_feature - 1] <= self.threshold:
                return self.right._take_decisions_instance(instance)
            else:
                return self.left._take_decisions_instance(instance)
        elif self.operator == OperatorCondition.LT:
            if instance.iloc[self.id_feature - 1] < self.threshold:
                return self.right._take_decisions_instance(instance)
            else:
                return self.left._take_decisions_instance(instance)
        elif self.operator == OperatorCondition.EQ:
            if instance.iloc[self.id_feature - 1] == self.threshold:
                return self.right._take_decisions_instance(instance)
            else:
                return self.left._take_decisions_instance(instance)
        elif self.operator == OperatorCondition.NEQ:
            if instance.iloc[self.id_feature - 1] != self.threshold:
                return self.right._take_decisions_instance(instance)
            else:
                return self.left._take_decisions_instance(instance)
        else:
            raise NotImplementedError("The operator " + str(self.operator) + " is not implemented.")
