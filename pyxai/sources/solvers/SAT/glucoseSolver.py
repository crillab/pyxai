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
        print("theory_cnf:", theory_cnf)
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
        
        def _symplify_theory(node, *, stack, parent, come_from, root):
            if node.is_leaf():
                return root
            id_feature = node.id_feature 
            left_consistent, right_consistent = is_node_consistent(node, stack)
            if left_consistent:
                # The left part is consistent, simplify recursively
                root = _symplify_theory(node.left, stack=stack + [-id_feature], parent=node, come_from=0, root=root)
            else:
                # The left part is inconsistent, replace this node with the right
                if come_from == None:
                    #The root change
                    root = _symplify_theory(node.right, stack=stack + [id_feature], parent=None, come_from=None, root=node.right)
                elif come_from == 0:
                    #Replace the node
                    parent.left = node.right
                    root = _symplify_theory(node.right, stack=stack + [id_feature], parent=parent, come_from=0, root=root)
                elif come_from == 1:
                    parent.right = node.right
                    root = _symplify_theory(node.right, stack=stack + [id_feature], parent=parent, come_from=1, root=root)

            if right_consistent:
                # The right part is consistent, simplify recursively
                root = _symplify_theory(node.right, stack=stack + [id_feature], parent=node, come_from=1, root=root)
            else:
                # The right part is inconsistent, replace with the left
                if come_from == None:
                    #The root change
                    root = _symplify_theory(node.left, stack=stack + [-id_feature], parent=None, come_from=None, root=node.left)
                elif come_from == 0:
                    #Replace the node
                    parent.left = node.left
                    root = _symplify_theory(node.left, stack=stack + [-id_feature], parent=parent, come_from=0, root=root)
                elif come_from == 1:
                    parent.right = node.left
                    root = _symplify_theory(node.left, stack=stack + [-id_feature], parent=parent, come_from=1, root=root)
        
            return root
        
        decision_tree.root = _symplify_theory(decision_tree.root, stack=[], parent=None, come_from=None, root=decision_tree.root)
        return decision_tree