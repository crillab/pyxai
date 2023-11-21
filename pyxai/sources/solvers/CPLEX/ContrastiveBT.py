from ortools.linear_solver import pywraplp
from pyxai.sources.core.structure.type import TypeLeaf


class ContrastiveBT:
    def __init__(self):
        pass


    def create_model_and_solve(self, explainer, theory, *, time_limit=None):
        forest = explainer._boosted_trees.forest
        leaves = [tree.get_leaves() for tree in forest]
        bin_len = len(explainer.binary_representation)
        solver = pywraplp.Solver.CreateSolver("SCIP")

        # Model variables
        x = [solver.BoolVar(f"x[{i}]") for i in range(bin_len)]  # The instance

        y = []
        for j, tree in enumerate(forest):
            y.append([solver.BoolVar(f"y[{j}][{i}]") for i in range(len(tree.get_leaves()))])  # Actives leaves

        z = [solver.BoolVar(f"z[{i}]") for i in range(bin_len)]  # The flipped variables

        # Constraints related to tree structure

        for j, tree in enumerate(forest):
            for i, leave in enumerate(tree.get_leaves()):
                t = TypeLeaf.LEFT if leave.parent.left == leave else TypeLeaf.RIGHT
                cube = forest[j].create_cube(leave.parent, t)
                nb_neg = sum((1 for l in cube if l < 0))
                nb_pos = sum((1 for l in cube if l > 0))
                constraint = solver.RowConstraint(-solver.infinity(), nb_neg)
                constraint.SetCoefficient(y[j][i], nb_pos + nb_neg)
                # print(cube)
                for l in cube:
                    constraint.SetCoefficient(x[abs(l) - 1], -1 if l > 0 else 1)

        # Only one leave activated per tree
        for j, tree in enumerate(forest):
            constraint = solver.RowConstraint(1, 1)
            for v in y[j]:
                constraint.SetCoefficient(v, 1)

        # Change the prediction
        if explainer.target_prediction == 1:
            constraint_target = solver.RowConstraint(-solver.infinity(), 0)
        else:
            constraint_target = solver.RowConstraint(0, solver.infinity())
        for j, tree in enumerate(forest):
            for i, leave in enumerate(tree.get_leaves()):
                constraint_target.SetCoefficient(y[j][i], leave.value)

        # Constraints related to theory
        if theory is not None:
            print(theory)
            for clause in theory:
                constraint = solver.RowConstraint(1, solver.infinity())
                for l in clause:
                    print(1)
                    constraint.SetCoefficient(x[abs(l) - 1], -1 if l < 0 else 1)


        # links between x and z
        for i in range(bin_len):
            const1 = solver.RowConstraint(-solver.infinity(), 1 if explainer.binary_representation[i] > 0 else 0)
            const1.SetCoefficient(x[i], 1)
            const1.SetCoefficient(z[i], -1)
            const2 = solver.RowConstraint(-solver.infinity(), -1 if explainer.binary_representation[i] > 0 else 0)
            const2.SetCoefficient(x[i], -1)
            const2.SetCoefficient(z[i], -1)

        # Objective function

        objective = solver.Objective()
        for i in range(bin_len):
            objective.SetCoefficient(z[i], 1)
        objective.SetMinimization()

        # Solve the problem

        #print(solver.ExportModelAsLpFormat(obfuscated=False))
        r= solver.Solve()
        print(r)
        # for v in solver.variables():
        #    print(v.name(), v.solution_value())
        return [explainer.binary_representation[i] for i in range(len(z)) if z[i].solution_value() > 0]
