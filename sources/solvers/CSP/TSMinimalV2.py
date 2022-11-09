import os
from pycsp3 import *

from pyxai.sources.core.tools.utils import flatten
from pycsp3.dashboard import options

OpOverrider = protect()


class TSMinimal():

    def __init__(self):
        pass


    def weight_float_to_int(self, weight):
        return int(weight * pow(10, 9))


    def get_leaves(self, trees):
        leaves = []
        for tree in trees:
            leaves_current_tree = []
            if tree.is_leaf():
                leaves_current_tree.append((tree.root, tree.root, self.weight_float_to_int(tree.root.value)))
            else:
                nodes = tree.compute_nodes_with_leaves(tree.root)
                for node in nodes:
                    if node.right.is_leaf():
                        leaves_current_tree.append((node.right, node, self.weight_float_to_int(node.right.value)))
                    if node.left.is_leaf():
                        leaves_current_tree.append((node.left, node, self.weight_float_to_int(node.left.value)))
            leaves.append(leaves_current_tree)
        return leaves


    def get_special_values(self, nTrees, leaves, n_classes, prediction, data_classes):
        if n_classes > 2:  # multi-classes
            specialValues = []
            for t in range(nTrees):
                if data_classes[t] == prediction:
                    # we have to deal with the minimum constraint, so, to avoid a leaf value, we take the max of the values + 1
                    specialValues.append(max([element[2] for element in leaves[t]]) + 1)
                else:
                    # we have to deal with the maximum constraint, so, to avoid a leaf value, we take the min of the values - 1
                    specialValues.append(min([element[2] for element in leaves[t]]) - 1)
            return specialValues
        else:  # 2-classes
            if prediction == 1:
                # we have to deal with the minimum constraint, so, to avoid a leaf value, we take the max of the values + 1
                return [max([element[2] for element in leaves[i]]) + 1 for i in range(nTrees)]
            else:
                # we have to deal with the maximum constraint, so, to avoid a leaf value, we take the min of the values - 1
                return [min([element[2] for element in leaves[i]]) - 1 for i in range(nTrees)]


    def create_model_minimal_abductive_BT(self, implicant, BTs, prediction, n_classes, implicant_id_features, from_reason):

        trees = BTs.forest
        BTs.reduce_trees(implicant, prediction)
        idVariableToLiteral = {abs(v): v for v in implicant}
        literalToPosition = {v: i for i, v in enumerate(implicant)}
        nVariables = len(implicant)
        map_id_features = {feature: [i for i, v in enumerate(implicant) if implicant_id_features[i] == feature] for feature in implicant_id_features}
        nTrees = len(trees)
        leaves = self.get_leaves(trees)
        data_classes = [tree.target_class for tree in trees]
        specialValues = self.get_special_values(nTrees, leaves, n_classes, prediction, data_classes)
        maxNLeaves = max([len(leaves[i]) for i in range(nTrees)])
        self.warm_start_file = None
        if from_reason is not None:
            warm_start = ["1" if v in from_reason else "0" for v in implicant]  #  for the 's' variables
            warm_start.extend(["*"] * nTrees * maxNLeaves)  #  for the 'w' variables
            warm_start.extend(["*"] * nTrees)  #  for the 'm' variables
            self.warm_start_file = "warm_start.txt"
            if os.path.exists(self.warm_start_file):
                os.remove(self.warm_start_file)
            f = open(self.warm_start_file, "a")
            f.write(" ".join(warm_start))
            f.close()

            # To clear a old model
        clear()

        # say if a literal of the implicant is activated (1) or not (0)
        s = VarArray(size=nVariables, dom={0, 1})

        # one variable on each leaf to say if the value of the leaf is involved in the Minimum/Maximum constraint or not
        w = VarArray(size=[nTrees, maxNLeaves],
                     dom=lambda i, j: {leaves[i][j][2], specialValues[i]} if j < len(leaves[i]) else {specialValues[i]})

        #  the weight of each tree computed thank to the Minimum/Maximum constraint
        m = VarArray(size=nTrees, dom=lambda i: {leaves[i][j][2] for j in range(len(leaves[i]))})


        def create_intension(i, j):

            OpOverrider.enable()
            dt = trees[i]
            leaf = leaves[i][j][0]
            node = leaves[i][j][1]
            weight = leaves[i][j][2]
            if leaf == node:
                # special case where a leaf is the root, there is no condition !
                return (w[i][j] == weight)

            id_variable = dt.get_id_variable(node)
            literal = idVariableToLiteral[id_variable]
            position = literalToPosition[literal]

            go_right_or_left = 1 if node.right == leaf else 0  # right:0 #left:1
            sign_literal = 1 if literal > 0 else 0

            condition = []

            if go_right_or_left != sign_literal:
                condition.append(s[position] == 0)

            previous = node
            node = node.parent

            while node is not None:
                id_variable = dt.get_id_variable(node)
                literal = idVariableToLiteral[id_variable]
                position = literalToPosition[literal]
                go_right_or_left = 1 if node.right == previous else 0
                sign_literal = 1 if literal > 0 else 0

                if go_right_or_left != sign_literal:
                    condition.append(s[position] == 0)
                previous = node
                node = node.parent
            if condition == []:  # special case where the leave is alway activated
                return (w[i][j] == weight)
            condition = conjunction(condition)
            if_part = (w[i][j] == weight)
            else_part = (w[i][j] == specialValues[i])

            OpOverrider.disable()
            return ift(condition, if_part, else_part)


        if implicant_id_features != []:
            # if we are in the ReasonExpressivity.Features mode
            satisfy(
                [AllEqual(s[i] for i in map_id_features[feature]) for feature in map_id_features.keys() if len(map_id_features[feature]) > 1]
            )

        satisfy(
            [create_intension(i, j) for i in range(nTrees) for j in range(len(leaves[i]))]
        )

        if n_classes > 2:  # multi-classes case
            satisfy(
                [m[t] == Minimum(w[t]) for t in range(nTrees) if data_classes[t] == prediction],
                [m[t] == Maximum(w[t]) for t in range(nTrees) if data_classes[t] != prediction],
                [Sum(m[t] for t in range(nTrees) if data_classes[t] == prediction) > Sum(
                    m[t] for t in range(nTrees) if data_classes[t] == other_class) for other_class in set(data_classes) if other_class != prediction]
            )
        else:  # 2-classes case
            if prediction == 1:
                satisfy(
                    [m[t] == Minimum(w[t]) for t in range(nTrees)],
                    Sum(m) > 0,
                )
            else:
                satisfy(
                    [m[t] == Maximum(w[t]) for t in range(nTrees)],
                    Sum(m) < 0,
                )

        if implicant_id_features != []:
            # if we are in the ReasonExpressivity.Features mode
            minimize(Sum(s[map_id_features[feature][0]] for feature in map_id_features.keys()))
        else:
            minimize(Sum(s))

        # keep trees in initial state
        BTs.remove_reduce_trees()


    def solve(self, time_limit=None, upper_bound=-1):

        ace = solver(ACE)
        options.output = "/tmp"
        t = " -t=" + str(time_limit) + "s" if time_limit is not None else ""
        warm = " -warm=" + str(self.warm_start_file) if self.warm_start_file is not None else ""
        ub = "" if upper_bound == -1 else f" -ub={upper_bound}"

        ace.setting("-ale=4 -di=0" + t + warm + ub)
        instance = compile()
        result = ace.solve(instance, verbose=False)

        if result == OPTIMUM or result == SAT:
            return result, [value for i, value in enumerate(solution().values) if
                            solution().variables[i] is not None and 's' in solution().variables[i].id]

        return result, []
