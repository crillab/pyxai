from ortools.linear_solver import pywraplp
from .help_functions import *


class Range:
    def __init__(self):
        pass


    def create_model_and_solve(self, explainer, theory, partial_binary_representation, minimum, time_limit):
        forest = explainer._boosted_trees.forest
        bin_len = len(explainer.binary_representation)
        solver = pywraplp.Solver.CreateSolver("SCIP")
        features_to_bin = explainer._boosted_trees.get_id_binaries()

        if time_limit is not None:
            solver.SetTimeLimit(time_limit * 1000)  # time limit in milisecond

        # Model variables
        instance = instance_variables(solver, bin_len)           # The instance
        active_leaves = active_leaves_variables(solver, forest)  # active leaves


        # Constraints related to tree structure
        tree_structure_constraints(explainer, solver, active_leaves, instance)

        # Constraints related to theory
        theory_constraints(solver, instance, theory)

        # Fix partial binary representation
        for lit in partial_binary_representation:
            v = 1 if lit > 0 else 0
            constraint = solver.RowConstraint(v, v)
            constraint.SetCoefficient(instance[abs(lit) - 1], 1)

        # set the objective
        objective = solver.Objective()

        for j, tree in enumerate(forest):
            for i, leave in enumerate(tree.get_leaves()):
                objective.SetCoefficient(active_leaves[j][i], leave.value)

        if minimum:
            objective.SetMinimization()
        else:
            objective.SetMaximization()

        solver.Solve()
        return solver.Objective().Value()