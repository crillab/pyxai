from pysat.formula import WCNF


class MAXSATSolver:
    def __init__(self):
        self.WCNF = WCNF()


    def add_soft_clause(self, clause, weight):
        self.WCNF.append(clause, weight=weight)


    def add_hard_clause(self, clause):
        self.WCNF.append(clause)


    def solve(self, *, time_limit=0):
        raise NotImplementedError
