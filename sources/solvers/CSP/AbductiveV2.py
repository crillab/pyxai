import os
from pycsp3 import *

from pyxai.sources.core.tools.utils import flatten

OpOverrider = protect()


class AbductiveModelV2():

    def __init__(self):
        pass


    def weight_float_to_int(self, weight):
        return int(weight * pow(10, 9))


    def get_leaves(self, BTs, implicant, current_removed, is_removed):
        leaves = []
        trees = BTs.forest
        for tree in trees:
            leaves_current_tree = leaves_current_tree.append(tree.root) if tree.is_leaf() else BTs.get_leaves(tree, tree.root, implicant, is_removed,
                                                                                                              current_removed, False)
            leaves.append(leaves_current_tree)
        return leaves


    def create_model_is_abductive(self, implicant, current_removed, is_removed, BTs, prediction):
        # This model check if no complete implicant are misclassified.
        # UNSAT -> no complete implicant are misclassified -> it is abductive
        #  SAT -> a model is misclassified -> it is not abductive
        # implicant: the complete implicant (representing the instance)
        # is_removed: say if implicant[i] is removed or not
        # BTs: the model
        #  prediction: the prediction to have.
        #  n_classes: the number of classes.

        # To clear a old model
        clear()

        trees = BTs.forest
        nTrees = len(trees)
        n_classes = BTs.n_classes
        classes = [tree.target_class for tree in trees]
        removed_indexes = [i for i, _ in enumerate(implicant) if is_removed[i]]
        leaves = self.get_leaves(BTs, implicant, current_removed, is_removed)
        # print("leaves:", leaves)

        weights = [[self.weight_float_to_int(leave.value) for (leave, _) in tmp_leaves] for tmp_leaves in leaves]

        # print("implicant:", implicant)
        print("is_removed:", is_removed)
        # print("indexes_is_removed:", removed_indexes)

        # print("weights:", weights)
        # print("nClasses:", n_classes)
        idVariableToLiteral = {abs(v): v for v in implicant}
        literalToPositionInImplicant = {v: i for i, v in enumerate(implicant)}  # Position of the literal in implicant
        literalToPositionInRemovedIndexes = {implicant[v]: i for i, v in enumerate(removed_indexes)}

        #  To go on the leaf or the right side of a condition for the removed literals. Allow to explore all possible weights.
        d = VarArray(size=len(removed_indexes), dom={0, 1})

        # The weight of each tree
        w = VarArray(size=nTrees, dom=lambda i: weights[i])


        def create_intension(i, j):

            OpOverrider.enable()
            dt = trees[i]  # The tree
            leaf = leaves[i][j][0]  # The leaf node
            is_in_current = leaves[i][j][1]
            weight = self.weight_float_to_int(leaf.value)  #  The weight

            condition = []  #  The condition part of the constraint to build

            node = leaf.parent  #  The node that we start, so, from now, node is not a leaf
            previous = leaf  #  The previous node

            while node is not None:
                id_variable = dt.get_id_variable(node)
                literal = idVariableToLiteral[id_variable]
                position = literalToPositionInImplicant[literal]
                if is_removed[position]:
                    # this literal is removed
                    go_right_or_left = 1 if node.right == previous else 0  # right:0 #left:1
                    good_position = literalToPositionInRemovedIndexes[literal]
                    condition.append(d[good_position] == go_right_or_left)
                previous = node
                node = node.parent

            if condition == []:  # special case where there is only one leaf (i.e. no removed literal in the nodes)
                return (w[i] == weight)

            condition = conjunction(condition)
            if_part = (w[i] == weight)

            OpOverrider.disable()
            if not is_in_current:
                return None
            return imply(condition, if_part)


        satisfy(
            [create_intension(i, j) for i in range(nTrees) for j in range(len(leaves[i]))]
        )
        if n_classes > 2:  # multi-classes case
            pass  # TODO
        else:  # 2-classes case
            if prediction == 1:
                satisfy(Sum(w) < 0)  # We invert to find UNSAT (A solution that do not predict the good class)
            else:
                satisfy(Sum(w) > 0)  # We invert to find UNSAT (A solution that do not predict the good class)


    def solve(self, time_limit=0):

        ace = solver(ACE)
        t = " -t=" + str(time_limit) + "s" if time_limit != 0 else ""
        ace.setting("-ale=4 -di=0 -valh=Last" + t)
        instance = compile()
        result = ace.solve(instance, verbose=True)

        # if result == OPTIMUM or result == SAT:
        #  return result, [value for i, value in enumerate(solution().values) if solution().variables[i] is not None and 's' in solution().variables[i].id]

        return result, []
