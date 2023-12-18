import os
import subprocess
import time
import uuid

from pyxai import Explainer
from pyxai.sources.core.tools.utils import get_os

OPENWBO_DIRECTORY = os.sep.join(__file__.split(os.sep)[:-1]) + os.sep
OPENWBO_EXEC = OPENWBO_DIRECTORY + "openwbo-" + get_os()


class EncoreSolver():
    def __init__(self, reference_instances, _hash=str(uuid.uuid4().fields[-1])[:8]):
        super().__init__()
        self._hash = _hash
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        self.reference_instances_filenames = ["tmp" + os.sep + "encore-" + self._hash + "-reference-instances-" + str(label) +".cnf" for label in reference_instances.keys()]
        self.model_filename = "tmp" + os.sep + "encore-" + self._hash + "-model.wcnf"

    def solve(self, *, time_limit=0):
        return None
