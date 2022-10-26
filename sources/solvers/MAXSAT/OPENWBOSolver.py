import os
import subprocess
import time
import uuid

from pyxai import Explainer
from pyxai.sources.core.tools.utils import get_os
from pyxai.sources.solvers.MAXSAT.MAXSATSolver import MAXSATSolver

OPENWBO_DIRECTORY = os.sep.join(__file__.split(os.sep)[:-1]) + os.sep
OPENWBO_EXEC = OPENWBO_DIRECTORY + "openwbo-" + get_os()


class OPENWBOSolver(MAXSATSolver):
    def __init__(self, _hash=str(uuid.uuid4().fields[-1])[:8]):
        super().__init__()
        self._hash = _hash
        self.filename_wcnf = "/tmp/wbo-" + self._hash + ".wcnf"


    def solve(self, *, time_limit=0):
        wcnf_file = f"/tmp/wbo-{self._hash}.wcnf"
        self.WCNF.to_file(wcnf_file)
        params = [OPENWBO_EXEC]
        time_used = -time.time()
        if time_limit != 0:
            params += [f"-cpu-lim={time_limit}"]
        params += [wcnf_file]
        p = subprocess.run(params, timeout=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        time_used += time.time()
        output_str = [line.split(" ") for line in p.stdout.split("\n") if len(line) > 0 and line[0] == "v"]
        if len(output_str) == 0:
            return p.stderr, None, time_used

        status = [line.split(" ")[1] for line in p.stdout.split("\n") if len(line) > 0 and line[0] == "s"][0]
        model = [int(lit) for lit in output_str[0] if lit != 'v' and lit != '']
        return status, model, Explainer.TIMEOUT if status == "SATISFIABLE" else time_used
