from ortools.linear_solver import pywraplp
from pyxai.sources.core.explainer.Explainer import Explainer
from .help_functions import *

class ContrastiveBT:
    def __init__(self):
        pass


    def create_model_and_solve(self, explainer, theory, excluded, n, time_limit):
        forest = explainer._boosted_trees.forest

        bin_len = len(explainer.binary_representation)
        solver = pywraplp.Solver.CreateSolver("SCIP")
        features_to_bin = explainer._boosted_trees.get_id_binaries()

        if time_limit is not None:
            solver.SetTimeLimit(time_limit * 1000)  # time limit in milisecond

        # Model variables
        instance = instance_variables(solver, bin_len)           # The instance
        active_leaves = active_leaves_variables(solver, forest)  # active leaves


        flipped = [solver.BoolVar(f"z[{i}]") for i in range(bin_len)]  # The flipped variables

        # Constraints related to tree structure
        tree_structure_constraints(explainer, solver, active_leaves, instance)

        # Constraints related to theory
        theory_constraints(solver, instance, theory)

        # Change the prediction
        if explainer.target_prediction == 1:
            constraint_target = solver.RowConstraint(-solver.infinity(), 0)
        else:
            constraint_target = solver.RowConstraint(0, solver.infinity())
        for j, tree in enumerate(forest):
            for i, leave in enumerate(tree.get_leaves()):
                constraint_target.SetCoefficient(active_leaves[j][i], leave.value)


        # links between instance and flipped
        for i in range(bin_len):
            const1 = solver.RowConstraint(-solver.infinity(), 1 if explainer.binary_representation[i] > 0 else 0)
            const1.SetCoefficient(instance[i], 1)
            const1.SetCoefficient(flipped[i], -1)
            const2 = solver.RowConstraint(-solver.infinity(), -1 if explainer.binary_representation[i] > 0 else 0)
            const2.SetCoefficient(instance[i], -1)
            const2.SetCoefficient(flipped[i], -1)

        # Set excluded features
        for lit in excluded:
            constraint = solver.RowConstraint(0, 0)
            constraint.SetCoefficient(flipped[abs(lit) - 1], 1)


        if theory is None: # the same encoding for RF : if theory minimal wrt features else wrt bin...
        # TODO : let the possibilit for the user to choose
            # Objective function
            objective = solver.Objective()
            for i in range(bin_len):
                objective.SetCoefficient(flipped[i], 1)
            objective.SetMinimization()
        else:
            # links between features and flipped
            dist_features = [solver.BoolVar(f"fd{i}") for i in range(len(features_to_bin))]
            i = 0
            for f, binaries in features_to_bin.items():
                constraint = solver.RowConstraint(-solver.infinity(), 0)
                constraint.SetCoefficient(dist_features[i], -1)
                for lit in binaries:
                    constraint.SetCoefficient(flipped[abs(lit -1)], 1 / len(binaries))
                i = i + 1
            # Objective function
            objective = solver.Objective()
            for d in dist_features:
                objective.SetCoefficient(d, 1)
            objective.SetMinimization()


        # print(solver.ExportModelAsLpFormat(obfuscated=False))

        # Solve the problem and extract n solutions
        results = []
        first = True
        best_objective = -1
        while True:
            if first:
                status = solver.Solve()
            else:
                status = solver.NextSolution()
            if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
                break
            solution = [explainer.binary_representation[i] for i in range(len(flipped)) if flipped[i].solution_value() >= 0.5]
            if first:
                best_objective = len(solution)
            first = False
            if len(solution) > best_objective:
                break
            results.append(solution)
            if len(results) == n:
                break
        return Explainer.format(results, n)