from pyxai.sources.core.structure.type import TypeLeaf
from pyxai.sources.core.structure.type import TypeLeaf


def instance_variables(solver, bin_len):
    return [solver.BoolVar(f"x[{i}]") for i in range(bin_len)]

def  active_leaves_variables(solver, forest):
    active_leaves = []
    for j, tree in enumerate(forest):
        active_leaves.append([solver.BoolVar(f"y[{j}][{i}]") for i in range(len(tree.get_leaves()))])  # Actives leaves
    return active_leaves


def tree_structure_constraints(explainer, solver, active_leaves, instance):
    forest = explainer._boosted_trees.forest
    for j, tree in enumerate(forest):
        for i, leave in enumerate(tree.get_leaves()):
            t = TypeLeaf.LEFT if leave.parent.left == leave else TypeLeaf.RIGHT
            cube = forest[j].create_cube(leave.parent, t)
            nb_neg = sum((1 for l in cube if l < 0))
            nb_pos = sum((1 for l in cube if l > 0))
            constraint = solver.RowConstraint(-solver.infinity(), nb_neg)
            constraint.SetCoefficient(active_leaves[j][i], nb_pos + nb_neg)
            for lit in cube:
                constraint.SetCoefficient(instance[abs(lit) - 1], -1 if lit > 0 else 1)

    # Only one leave activated per tree
    for j, tree in enumerate(forest):
        constraint = solver.RowConstraint(1, 1)
        for v in active_leaves[j]:
            constraint.SetCoefficient(v, 1)


def theory_constraints(solver, instance, theory):
    if theory is not None:
        for clause in theory:
            constraint = solver.RowConstraint(-solver.infinity(), 0)
            for l in clause:
                constraint.SetCoefficient(instance[abs(l) - 1], 1 if l < 0 else -1)
