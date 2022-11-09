import time

from pysat.examples.rc2 import RC2
from pysat.pb import *

from pyxai.sources.solvers.MAXSAT.MAXSATSolver import MAXSATSolver


# from pysat.pb import PBEnc

class RC2MAXSATsolver(MAXSATSolver):

    def __init__(self):
        super().__init__()


    def add_soft_clauses_implicant(self, implicant):
        for lit in implicant:
            self.WCNF.append([-lit], weight=1)


    def add_hard_clauses(self, tree_CNF, implicant=None):
        for clause in tree_CNF:  # hard
            if implicant is not None:
                new_clause = [lit for lit in clause if lit in implicant]
                assert new_clause != [], "This clause cannot be empty"
                self.WCNF.append(new_clause)
            else:
                self.WCNF.append(clause)


    def add_atmost(self, implicant, sufficient_reason):
        self.WCNF.extend(
            PBEnc.atmost(lits=implicant, top_id=self.WCNF.nv + 1, bound=len(sufficient_reason)).clauses)


    def add_clause(self, clause):
        self.WCNF.append(clause)


    def solve_implicant(self, implicant):
        result = RC2(self.WCNF).compute()
        return None if result is None else [lit for lit in result if lit in implicant]


    def solve(self, *, time_limit=0):
        time_used = - time.time()
        result = RC2(self.WCNF).compute()
        time_used += time.time()
        return (None, None, None) if result is None else ("OPTIMUM FOUND", [lit for lit in result], time_used)
