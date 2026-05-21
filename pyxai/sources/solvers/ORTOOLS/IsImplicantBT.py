

from ortools.linear_solver import pywraplp
from pyxai.sources.type import TypeLeaf
import numpy

class IsImplicantBT():

    def __init__(self, _boosted_trees, binary_instance, implicant, target_prediction, theory, base_score):
        self._boosted_trees = _boosted_trees # The boosted trees model
        self.binary_instance = binary_instance # The complete instance in binary representation (literals)
        self.implicant = implicant # A incomplete implicant (literals forming the explanation)
        self.target_prediction = target_prediction # The target prediction we want to check
        self.n_classes = self._boosted_trees.n_classes
        self.theory = theory
        self.base_score = base_score
        #print("self.theory:", self.theory)
        #print("self._boosted_trees:", self._boosted_trees)
        #print("self.binary_instance:", self.binary_instance)
        #print("self.implicant:", self.implicant)
        #print("self.target_prediction:", self.target_prediction)
        #print("self.n_classes:", self.n_classes)
        
        # Shortcuts 
        self.trees = self._boosted_trees.forest
        self.n_trees = len(self.trees)
        self.leaves = [tree.get_leaves() for tree in self.trees]
        #print("n_leaves per tree:", [len(self.leaves[i]) for i in range(len(self.leaves))])

    def is_implicant(self):
        # Create the MIP solver with the CBC backend.
        
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            raise Exception("Solver not created.")
        # current_instance_{i} for a literal i of the instance, 1 if the associated condition in the trees is satisfied else 0. Allow to explore all possible paths int the trees.
        self.current_instance = [solver.BoolVar(f"current_instance_{i}") for i in range(1 + len(self.binary_instance))]
        
        # leaf_{i,j} is true if the leaf j of the tree i is active (the current  goes to this leaf in the tree i)
        self.leaf = [[solver.BoolVar(f"leaf_" + str(i) + "_"+ str(j)) for j in range(len(self.leaves[i]))] for i in range(len(self.leaves))] # nb leaves

        # Value of each tree
        self.tree_weights = [solver.NumVar(self._boosted_trees.forest[i].get_min_value(), self._boosted_trees.forest[i].get_max_value(), "tree_weights" + str(i)) for i in range(self.n_trees)]

        # Value of the forest
        self.forest_weight = solver.NumVar(-solver.infinity(), solver.infinity(), "forest_weight")

        # Constraints 
        # Set the current_instance variables according to the implicant
        for lit in self.implicant:
            if lit > 0:
                ct = solver.RowConstraint(1, 1)
                ct.SetCoefficient(self.current_instance[lit], 1)
            else:
                ct = solver.RowConstraint(0, 0)
                ct.SetCoefficient(self.current_instance[-lit], 1)
        
        # Add the constraints for the theory
        # if self.theory is not None:
        #     for clause in self.theory:
        #         if clause[0] < 0 and clause[1] < 0:
        #             # Categorial feature
        #             constraint = solver.RowConstraint(-solver.infinity(), 1)
        #             for l in clause:
        #                 constraint.SetCoefficient(self.current_instance[abs(l)], 1)
        #         elif clause[0] < 0 and clause[1] > 0:
        #             # Numerical feature
        #             constraint = solver.RowConstraint(0, solver.infinity())
        #             constraint.SetCoefficient(self.current_instance[abs(clause[0])], -1)
        #             constraint.SetCoefficient(self.current_instance[abs(clause[1])], 1)
        #         else:   
        #             raise NotImplementedError("Not implemented yet for this kind of theory.")


        for i in range(self.n_trees):
            #print("Tree ", i)
            # (4): Only one leaf can be active in each tree
            ct1 = solver.RowConstraint(1, 1)
            for j in range(len(self.leaves[i])):
                ct1.SetCoefficient(self.leaf[i][j], 1)

            # (5): The value of the tree is the value of the active leaf
            ct2 = solver.RowConstraint(0, 0)
            for j in range(len(self.leaves[i])):
                ct2.SetCoefficient(self.leaf[i][j], self.leaves[i][j].value)
            ct2.SetCoefficient(self.tree_weights[i], -1)

            # (3):Constraints to link current_instance and leaf variables
            for j in range(len(self.leaves[i])):
                leaf = self.leaves[i][j]

                if leaf.parent is None:
                    cts = solver.RowConstraint(self.leaves[i][j].value, self.leaves[i][j].value)
                    cts.SetCoefficient(self.tree_weights[i], 1)
                    break
            
                type_leaf = TypeLeaf.LEFT if leaf.parent.left == leaf else TypeLeaf.RIGHT
                cube = self.trees[i].create_cube(leaf.parent, type_leaf)
                #print("cube:", cube)
                nb_neg = sum((1 for l in cube if l < 0))
                nb_pos = sum((1 for l in cube if l > 0))
                #ct3 = solver.RowConstraint(-solver.infinity(), nb_neg)
                #ct3.SetCoefficient(self.leaf[i][j], nb_pos + nb_neg)
                #for l in cube:
                #   if l > 0:
                #       ct3.SetCoefficient(self.current_instance[l], -1)
                #   else:
                #       ct3.SetCoefficient(self.current_instance[-l], 1)

                ct3 = solver.RowConstraint(-solver.infinity(), nb_pos - 1)
                ct3.SetCoefficient(self.leaf[i][j], -1)
                for l in cube:
                     if l > 0:
                        ct3.SetCoefficient(self.current_instance[l], 1)
                     else:
                        ct3.SetCoefficient(self.current_instance[-l], -1)


        # (6): The value of the forest is the sum of the values of the trees
        base_score = -numpy.log(1 / self.base_score - 1) if self.base_score != 0 else 0
        ct4 = solver.RowConstraint(-base_score, -base_score) # We add the base_score on the left part of the constraint to ensure that it is taken into account in the forest weight, i.e., forest_weight = sum(tree_weights) + base_score
        for i in range(self.n_trees):
            ct4.SetCoefficient(self.tree_weights[i], 1)
        ct4.SetCoefficient(self.forest_weight, -1)  

        if self.n_classes == 2:
            if self.target_prediction == 0:
                ct5 = solver.RowConstraint(0, solver.infinity())
            else:
                ct5 = solver.RowConstraint(-solver.infinity(), 0)
            ct5.SetCoefficient(self.forest_weight, 1)
        else:
            raise NotImplementedError("Not implemented yet for multi-class boosted trees.")

        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            # print("Solution:")
            # print("self.current_instance:", [self.current_instance[i].solution_value() for i in range(len(self.current_instance))])
            
            # print("self.leaf:", [[self.leaf[i][j].solution_value() for j in range(len(self.leaves[i]))] for i in range(len(self.leaves))])
            # print("self.tree_weights:", [self.tree_weights[i].solution_value() for i in range(self.n_trees)])
            # print("self.forest_weight:", self.forest_weight.solution_value())
            # print("Another solution?", status)
            return False  # A feasible solution found, thus the explanation is not an implicant
        else:
            return True  # No feasible solution found, thus the explanation is an implicant
            
