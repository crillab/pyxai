import copy

from pyxai.sources.core.structure.type import OperatorCondition


class LeafNode:
    def __init__(self, value):
        self.value = value
        self.parent = None


    def is_implicant(self, reason, target_prediction, map_features_to_id_binaries):
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


    def take_decisions_binary_representation(self, binary_representation=None, map_features_to_id_binaries=None):
        return self.value


    def take_decisions_instance(self, instance=None):
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


class DecisionNode:
    """
    A decision node represent a decision. A decision tree consists of these nodes.
    """


    def __init__(self, id_feature, *, threshold=0.5, operator=OperatorCondition.GE, left, right, parent=None):
        """
        Allow to construct a decision node that do not a leaf
        """
        self.id_feature = id_feature
        self.threshold = threshold
        self.operator = operator
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
                    self.right = copy.deepcopy(other_tree.root)
            else:
                if self.right.value == 0:
                    self.right = copy.deepcopy(other_tree.root)
        else:
            self.right.concatenate_tree(other_tree, disjunction)

        if self.left.is_leaf():
            assert self.left.value == 1 or self.left.value == 0, "The concatenation of a tree is only possible with a 0 or 1 leaf value !"
            if not disjunction:
                if self.left.value == 1:
                    self.left = copy.deepcopy(other_tree.root)
            else:
                if self.left.value == 0:
                    self.left = copy.deepcopy(other_tree.root)

        else:
            self.left.concatenate_tree(other_tree, disjunction)


    def is_leaf(self):
        return self.artificial_leaf


    def __str__(self):
        return "f{}<{}".format(self.id_feature, self.threshold)


    def is_implicant(self, binary_representation, target_prediction, map_features_to_id_binaries):
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
            return self.right.is_implicant(binary_representation, target_prediction, map_features_to_id_binaries)
        elif -id_variable in binary_representation:
            return self.left.is_implicant(binary_representation, target_prediction, map_features_to_id_binaries)
        else:
            right = self.right.is_implicant(binary_representation, target_prediction, map_features_to_id_binaries)
            left = self.left.is_implicant(binary_representation, target_prediction, map_features_to_id_binaries)
            return right and left


    def take_decisions_binary_representation(self, binary_representation, map_features_to_id_binaries):
        """
        Return the prediction (the classification) of a binary representation according to this node.
        Warning: The binary representation (propositional formula calculated from a instance) have to be complete
         (i.e only a direct reason, not a reason)
        Warning: right nodes are considered as the 'yes' responses of conditions, left nodes as 'no'.
        """
        id_variable = map_features_to_id_binaries[(self.id_feature, self.operator, self.threshold)][0]
        assert (id_variable in binary_representation) or (-id_variable in binary_representation), "The binary representation has to be complete !"
        if id_variable in binary_representation:
            return self.right.take_decisions_binary_representation(binary_representation, map_features_to_id_binaries)
        else:
            return self.left.take_decisions_binary_representation(binary_representation, map_features_to_id_binaries)


    def take_decisions_instance(self, instance):
        """
        Return the prediction (the classification) of an observation (instance) according to this node.
        This return value is either 0 or 1: 0 for the first (boolean) prediction value, 1 for the second one.
        Warning: right nodes are considered as the 'yes' responses of conditions, left nodes as 'no'.
        """
        # print("self.id_feature:", self.id_feature)

        if self.operator == OperatorCondition.GE:
            if instance[self.id_feature - 1] >= self.threshold:
                return self.right.take_decisions_instance(instance)
            else:
                return self.left.take_decisions_instance(instance)
        elif self.operator == OperatorCondition.GT:
            if instance[self.id_feature - 1] > self.threshold:
                return self.right.take_decisions_instance(instance)
            else:
                return self.left.take_decisions_instance(instance)
        elif self.operator == OperatorCondition.LE:
            if instance[self.id_feature - 1] <= self.threshold:
                return self.right.take_decisions_instance(instance)
            else:
                return self.left.take_decisions_instance(instance)
        elif self.operator == OperatorCondition.LT:
            if instance[self.id_feature - 1] < self.threshold:
                return self.right.take_decisions_instance(instance)
            else:
                return self.left.take_decisions_instance(instance)
        elif self.operator == OperatorCondition.EQ:
            if instance[self.id_feature - 1] == self.threshold:
                return self.right.take_decisions_instance(instance)
            else:
                return self.left.take_decisions_instance(instance)
        elif self.operator == OperatorCondition.NEQ:
            if instance[self.id_feature - 1] != self.threshold:
                return self.right.take_decisions_instance(instance)
            else:
                return self.left.take_decisions_instance(instance)
        else:
            raise NotImplementedError("The operator " + str(self.operator) + " is not implemented.")
