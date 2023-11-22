from ortools.linear_solver import pywraplp
from pyxai.sources.core.structure.type import TypeLeaf


class ContrastiveBT:
    def __init__(self):
        pass


    def create_model_and_solve(self, explainer, theory, time_limit):
        forest = explainer._boosted_trees.forest
        leaves = [tree.get_leaves() for tree in forest]
        bin_len = len(explainer.binary_representation)
        solver = pywraplp.Solver.CreateSolver("SCIP")

        if time_limit is not None:
            solver.SetTimeLimit(time_limit * 1000)  # time limit in milisecond

        # Model variables
        instance = [solver.BoolVar(f"x[{i}]") for i in range(bin_len)]  # The instance

        active_leaves = []
        for j, tree in enumerate(forest):
            active_leaves.append([solver.BoolVar(f"y[{j}][{i}]") for i in range(len(tree.get_leaves()))])  # Actives leaves

        flipped = [solver.BoolVar(f"z[{i}]") for i in range(bin_len)]  # The flipped variables

        # Constraints related to tree structure
        for j, tree in enumerate(forest):
            for i, leave in enumerate(tree.get_leaves()):
                t = TypeLeaf.LEFT if leave.parent.left == leave else TypeLeaf.RIGHT
                cube = forest[j].create_cube(leave.parent, t)
                nb_neg = sum((1 for l in cube if l < 0))
                nb_pos = sum((1 for l in cube if l > 0))
                constraint = solver.RowConstraint(-solver.infinity(), nb_neg)
                constraint.SetCoefficient(active_leaves[j][i], nb_pos + nb_neg)
                for l in cube:
                    constraint.SetCoefficient(instance[abs(l) - 1], -1 if l > 0 else 1)

        # Only one leave activated per tree
        for j, tree in enumerate(forest):
            constraint = solver.RowConstraint(1, 1)
            for v in active_leaves[j]:
                constraint.SetCoefficient(v, 1)

        # Change the prediction
        if explainer.target_prediction == 1:
            constraint_target = solver.RowConstraint(-solver.infinity(), 0)
        else:
            constraint_target = solver.RowConstraint(0, solver.infinity())
        for j, tree in enumerate(forest):
            for i, leave in enumerate(tree.get_leaves()):
                constraint_target.SetCoefficient(active_leaves[j][i], leave.value)

        # Constraints related to theory
        if theory is not None:
            print(theory)
            for clause in theory:
                constraint = solver.RowConstraint(-solver.infinity(), 0)
                for l in clause:
                    constraint.SetCoefficient(instance[abs(l) - 1], 1 if l < 0 else -1)

        # links between instance and flipped
        for i in range(bin_len):
            const1 = solver.RowConstraint(-solver.infinity(), 1 if explainer.binary_representation[i] > 0 else 0)
            const1.SetCoefficient(instance[i], 1)
            const1.SetCoefficient(flipped[i], -1)
            const2 = solver.RowConstraint(-solver.infinity(), -1 if explainer.binary_representation[i] > 0 else 0)
            const2.SetCoefficient(instance[i], -1)
            const2.SetCoefficient(flipped[i], -1)

        # links between features and flipped


        # Objective function
        objective = solver.Objective()
        for i in range(bin_len):
            objective.SetCoefficient(flipped[i], 1)
        objective.SetMinimization()

        # Solve the problem
        # print(solver.ExportModelAsLpFormat(obfuscated=False))
        status = solver.Solve()
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return None
        return [explainer.binary_representation[i] for i in range(len(flipped)) if flipped[i].solution_value() >= 0.5]
