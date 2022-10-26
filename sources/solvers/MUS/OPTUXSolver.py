from pysat.examples.optux import OptUx
from pysat.formula import WCNF


class OPTUXSolver:

    def __init__(self):
        self.WCNF = WCNF()


    def add_soft_clauses(self, clauses, weight):
        for clause in clauses:
            self.WCNF.append(clause, weight=weight)


    def add_hard_clauses(self, clauses):
        for clause in clauses:
            self.WCNF.append(clause)


    def solve(self, implicant):
        minimal_len = len(implicant)
        reason = implicant
        with OptUx(self.WCNF) as optux:
            for mus in optux.enumerate():
                if len(mus) < minimal_len:
                    reason = [implicant[index - 1] for index in mus]  # enumerate return the indexes of soft clauses
                    minimal_len = len(mus)
        return reason
