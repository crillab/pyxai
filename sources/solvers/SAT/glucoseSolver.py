import time
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
