import time
import numpy
from threading import Timer

from pysat.solvers import Glucose4

def interrupt(s):
    s.interrupt()


class GlucoseSolver:

    def __init__(self):
        self.glucose = Glucose4()


    def add_clauses(self, clauses):
        self.glucose.append_formula(clauses)


    def solve(self, time_limit=None):
        time_used = -time.time()

        if time_limit is not None:
            timer = Timer(time_limit, interrupt, [self.glucose])
            timer.start()
            result = self.glucose.solve_limited(expect_interrupt=True)
        else:
            result = self.glucose.solve()
        time_used += time.time()
        return None if not result else self.glucose.get_model(), time_used

    def propagate(self, reason):
        return self.glucose.propagate(reason)
    
    def symplify_theory(self, decision_tree, theory_cnf):
        

        self.add_clauses(theory_cnf)
        
       
    
        def is_node_consistent(node, stack):
            if node.is_leaf():
                return True, True

            id_feature = node.id_feature    
            # Check consistency on the left
            stack.append(-id_feature)
            left_consistent = self.glucose.propagate(stack)[0]
            stack.pop()

            # Check consistency on the right
            stack.append(id_feature)
            right_consistent = self.glucose.propagate(stack)[0]
            stack.pop()

            return left_consistent, right_consistent
        
        def _symplify_theory(node, *, stack):
            if node.is_leaf():
                return node
            id_feature = node.id_feature 
            left_consistent, right_consistent = is_node_consistent(node, stack)

            if left_consistent:
                # The left part is consistent, simplify recursively
                left_simplified = _symplify_theory(node.left, stack=stack + [-id_feature])
            else:
                # The left part is inconsistent, replace with the right
                left_simplified = _symplify_theory(node.right, stack=stack + [id_feature])

            if right_consistent:
                # The right part is consistent, simplify recursively
                right_simplified = _symplify_theory(node.right, stack=stack + [id_feature])
            else:
                # The right part is inconsistent, replace with the left
                right_simplified = _symplify_theory(node.left, stack=stack + [-id_feature])    
            
            # If both sides are identical, simplify by replacing with either side
            raw = decision_tree.to_tuples(node)
            if raw[1] == raw[2]:
                print("equal:", raw[1], raw[2])
                return left_simplified
                
            node.left = left_simplified
            node.right = right_simplified
            return node

        return _symplify_theory(decision_tree.root, stack=[])