from random import random
from docplex.mp.model import Model
from pyxai.sources.core.structure.type import TypeLeaf
import time

class SufficientRegression:
    def __init__(self):
        pass


    def computeInterpretation(self, solution, varTrees: list):
        """
        Compute the interpretation computed by cplex.

        Attributes:
        ----------
        solution : Cplex solution
            is the solution returned by cplex
        varTrees : list
            is the set of variables.
        """
        interpretation = [0]
        for i in range(1, len(varTrees)):
            if solution[varTrees[i]] > 0.9:
                interpretation.append(i)
            else:
                interpretation.append(-i)
        return interpretation


    def create_model_and_solve(self, explainer, lb, ub, *, time_out=None):
        '''
        Model comes from IJCAI 23 paper. All constraitns of the model  are numbered as in this paper.
        '''
        #random.shuffle(featureBlock)
        verbose = True
        forest = explainer._boosted_trees.forest

        leaves = [tree.get_leaves() for tree in forest]


        extremum_range = explainer.extremum_range()
        model = Model()
        model.context.cplex_parameters.threads = 1

        var_trees = [model.binary_var("b_" + str(i)) for i in range(1 + len(explainer.binary_representation))] # max id binary var
        var_active_branches = [[model.binary_var("a_" + str(i) + "_"+ str(j)) for j in range(len(leaves[i]))] for i in range(len(leaves))] # nb leaves
        var_value_tree = [model.continuous_var(forest[i].get_min_value(), forest[i].get_max_value(), "t" + str(i)) for i in range(len(forest))]
        var_value_forest = model.continuous_var(lb=extremum_range[0], ub=extremum_range[1], name="forest")

        for i in range(len(forest)):
            model.add_constraint( # (4)
                model.sum(var_active_branches[i]) == 1
            )

            model.add_constraint( # (5)
                model.sum([var_active_branches[i][j] * leaves[i][j].value for j in range(len(leaves[i]))]) == var_value_tree[i]
            )

            for j in range(len(leaves[i])):
                leave = leaves[i][j]
                t = TypeLeaf.LEFT if  leave.parent.left == leave else TypeLeaf.RIGHT
                cube = forest[i].create_cube(leave.parent, t)
                model.add_constraint( # (3)
                    model.sum([var_trees[l] for l in cube if ( l > 0)] + [1 - var_trees[-l] for l in cube if (l < 0)] + [-var_active_branches[i][j]]) <= len(cube) - 1
                )

        model.add_constraint( # (6)
            model.sum(var_value_tree) == var_value_forest
        )

        varLb = model.binary_var("lb")
        varUb = model.binary_var("ub")

        model.add_indicator(varLb, var_value_forest <= lb, active_value=1) # (7)
        model.add_indicator(varUb, var_value_forest >= ub, active_value=1) # (8)
        model.add_constraint(varLb + varUb == 1)                           # (9)


        # add the category constraints
        '''for cat in listCatBin:
            for i in range(len(cat) - 1):
                for j in range(i + 1, len(cat)):
                    assert cat[i] < len(var_trees) and cat[j] < len(var_trees)
                    model.add_constraint(var_trees[cat[i]] + var_trees[cat[j]] <= 1)  # (1)

        # add the domain constraints.
        for feature in featureBlock:
            if len(feature) == 1:
                continue
            for i in range(len(feature) - 1):
                model.add_constraint(
                    var_trees[feature[i]] + 1 - var_trees[feature[i + 1]] >= 1)     # (1)
'''
        valuation = {}
        for i in explainer.binary_representation:
            valuation[abs(i)] = i > 0

        notKnown = []

        featureBlock = []
        for _ in range(len(explainer.instance)):
            featureBlock.append([])
        for i in range(len(explainer._implicant_id_features)):
            print(explainer._implicant_id_features[i] - 1)
            featureBlock[explainer._implicant_id_features[i] - 1].append(i)
        print(featureBlock, len(explainer.instance))


        for feature in featureBlock:
            tmp = []
            for l in feature:
                if abs(l) not in valuation:
                    continue
                if valuation[abs(l)]:
                    tmp.append((l, var_trees[l] == 1))
                else:
                    tmp.append((-l, var_trees[l] == 0))

                model.add_constraint(tmp[-1][1])
            notKnown.append(tmp)


        model.export_as_lp()

        nbRemoveByRotation = 0
        reason = []
        missclass = []

        starting_time = -time.process_time()

        timeCheck = False
        # by feature.
        if verbose:
            print("c First run, elimination by feature")
        # to visualize the model, it is in /tmp
        #model.export_as_lp()
        # exit(1)
        while (len(notKnown) > 0 and (time_out is None or (starting_time + time.process_time()) < time_out)):
            if time_out is not None:
                model.set_time_limit(max([1, time_out - (starting_time + time.process_time())]))
            oldnotKnown = notKnown + [] + reason
            if verbose:
                print(len(notKnown), len(reason))
            feature = notKnown[-1]
            notKnown.pop()

            # remove all the boolean related to the feature.
            for c in feature:
                model.remove(c[1])

            # check if it is a necessary feature.
            solution = model.solve()
            if solution is not None or (time_out is not None and (starting_time + time.process_time()) > time_out):
                # for the moment this feature is required.
                tmp = []
                for c in feature:
                    model.add_constraint(c[1])
                    tmp.append(c)
                reason.append(tmp)

        # by boolean.
        notKnown = reason
        # oldnotKnown = notKnown + []
        reason = []
        longReason = []

        if verbose:
            print("c Second run, elimination by boolean")

        while (len(notKnown) > 0 and (time_out is None or (starting_time + time.process_time()) < time_out)):
            if time_out is not None:
                model.set_time_limit(np.max([1, time_out - (starting_time + time.process_time())]))
            oldnotKnown = notKnown + []
            if verbose:
                print(notKnown, len(notKnown), len(reason))
            feature = notKnown[-1]
            notKnown.pop()

            # we already now that this feature has to be in the solution.
            if len(feature) == 1:
                reason.append(feature[0][0])
                longReason.append(feature[0][0])
                continue

            # search for the transition.
            posPoint = 0
            while (posPoint < len(feature) and feature[posPoint][0] > 0):
                posPoint += 1
            posPoint -= 1
            negPoint = posPoint + 1

            state = 0
            while (posPoint >= 0 or negPoint < len(feature) and (time_out is None or (starting_time + time.process_time()) < time_out)):
                if time_out is not None:
                    model.set_time_limit(np.max([1, time_out - (starting_time + time.process_time())]))
                ind = -1
                if posPoint < 0:
                    ind = negPoint
                    negPoint += 1
                elif negPoint >= len(feature):
                    ind = posPoint
                    posPoint -= 1
                elif state % 2 == 0:
                    ind = posPoint
                    posPoint -= 1
                else:
                    ind = negPoint
                    negPoint += 1
                state += 1

                current = feature[ind]
                model.remove(current[1])

                # check if it is a necessary boolean feature.
                solution = model.solve()
                if solution is not None:
                    # this boolean feature is required.
                    reason.append(current[0])
                    longReason.append(current[0])
                    model.add_constraint(current[1])

                    # we know that one the bound is fixed
                    if current[0] > 0:
                        while posPoint >= 0:
                            longReason.append(feature[posPoint][0])
                            posPoint -= 1
                    else:
                        while negPoint < len(feature):
                            longReason.append(feature[negPoint][0])
                            negPoint += 1

                    interpretation = self.computeInterpretation(solution, var_trees)
                    score = explainer.predict_implicant(interpretation)
                    if score > lb and score < ub:
                        missclass.append(current[0])

        if time_out is not None and (starting_time + time.process_time()) > time_out:
            timeCheck = True
            for f in oldnotKnown:
                posPoint = 0
                while (posPoint < len(f) and f[posPoint][0] > 0):
                    posPoint += 1
                posPoint -= 1
                negPoint = posPoint + 1
                if posPoint >= 0:
                    reason.append(f[posPoint][0])
                if negPoint < len(f):
                    reason.append(f[negPoint][0])
            reason = simplifyImplicant(reason, featureBlock, list_cat_bin=listCatBin)

        return reason, starting_time + time.process_time()

    def solve(self, time_limit=None):
        pass