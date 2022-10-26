import os
import subprocess
import time
import uuid

from pyxai import Explainer

MUSER_DIRECTORY = os.sep.join(__file__.split(os.sep)[:-1]) + os.sep
MUSER_EXEC = MUSER_DIRECTORY + "muser_static"


class MUSERSolver:
    def __init__(self, *, filenames="/tmp/muser-", _hash=str(uuid.uuid4().fields[-1])[:8]):
        self._hash = _hash
        self.filename_gcnf = filenames + self._hash + ".gcnf"


    def write_gcnf(self, n_variables, hard_clauses, soft_clauses):

        file = open(self.filename_gcnf, "w")
        file.write(f"p gcnf {n_variables} {len(hard_clauses) + len(soft_clauses)} {len(soft_clauses) + 1}\n")

        for clause in hard_clauses:
            file.write("{1} " + " ".join(str(lit) for lit in clause) + " 0\n")
        for i, clause in enumerate(soft_clauses):
            file.write("{" + str(i + 2) + "} " + " ".join(str(lit) for lit in clause) + " 0\n")

        file.close()


    def solve(self, time_limit=None):
        time_used = -time.time()
        try:
            p = subprocess.run([MUSER_EXEC, '-comp', '-grp', self.filename_gcnf], timeout=time_limit, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            time_used += time.time()
        except subprocess.TimeoutExpired:
            time_used = Explainer.TIMEOUT
        model = [line for line in p.stdout.split('\n') if line.startswith("v")][0]
        model = [int(lit) for lit in model.split(' ')[1:-1]]
        status = [line.split(" ")[1] for line in p.stdout.split("\n") if len(line) > 0 and line[0] == "s"][0]
        return model, status, time_used
