import time
import numpy
import copy
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
                return False, False
            id_literal = decision_tree.map_features_to_id_binaries[(node.id_feature, node.operator, node.threshold)][0]
            # Check consistency on the left
            stack.append(-id_literal)
            left_consistent = self.glucose.propagate(stack)[0]
            #if left_consistent is False:
            #    print("left_consistent:", stack)
            stack.pop()

            stack.append(id_literal)
            right_consistent = self.glucose.propagate(stack)[0]
            #if right_consistent is False:
            #    print("right_consistent:", stack)
            stack.pop()
            
            
            return left_consistent, right_consistent
        
        def _simplify_theory(node, *, stack, parent, come_from, root):
            #print("_symplify_theory start:", stack)
            if node.is_leaf():
                return root
            
            id_literal = decision_tree.map_features_to_id_binaries[(node.id_feature, node.operator, node.threshold)][0]
            left_consistent, right_consistent = is_node_consistent(node, stack)
            #print("left_consistent:", left_consistent)
            #print("right_consistent:", right_consistent)
            if left_consistent is False and right_consistent is False:
                #Impossible Case
                raise ValueError("Impossible Case : both are inconsistent")
            elif left_consistent is True and right_consistent is True:
                #Both consistent: continues the recurrence 
                root = _simplify_theory(node.left, stack=stack + [-id_literal], parent=node, come_from=0, root=root)
                root = _simplify_theory(node.right, stack=stack + [id_literal], parent=node, come_from=1, root=root)
                return root
            elif left_consistent is False:
                if come_from == None:
                    #The root change
                    root = _simplify_theory(node.right, stack=stack, parent=None, come_from=None, root=node.right)
                elif come_from == 0:
                    #Replace the node
                    #print("left inconsistent come from 0")
                    parent.left = node.right
                    return _simplify_theory(parent.left, stack=stack, parent=parent, come_from=0, root=root)
                elif come_from == 1:
                    #print("left inconsistent come from 1")
                    parent.right = node.right
                    return _simplify_theory(parent.right, stack=stack, parent=parent, come_from=1, root=root)
            elif right_consistent is False:
                if come_from == None:
                    #The root change
                    return _simplify_theory(node.left, stack=stack, parent=None, come_from=None, root=node.left)
                elif come_from == 0:
                    #Replace the node
                    #print("right inconsistent come from 0")
                    parent.left = node.left
                    return _simplify_theory(parent.left, stack=stack, parent=parent, come_from=0, root=root)
                elif come_from == 1:
                    #print("right inconsistent come from 1")
                    parent.right = node.left
                    return _simplify_theory(parent.right, stack=stack, parent=parent, come_from=1, root=root)
            else:
                raise ValueError("Impossible Case")
                
            return root
        
        decision_tree.root = _simplify_theory(decision_tree.root, stack=[], parent=None, come_from=None, root=decision_tree.root)
        return decision_tree