import os
import subprocess
import time
import uuid

from pyxai import Explainer
from pyxai.sources.core.tools.utils import get_os

DIRECTORY = os.sep.join(__file__.split(os.sep)[:-1]) + os.sep
ENCORE_DIRECTORY = DIRECTORY+".."+os.sep+".."+os.sep+".."+os.sep+".."+os.sep+".."+os.sep+"encore"+os.sep 
ENCORE_EXEC = ENCORE_DIRECTORY + "build"+os.sep+"encore"


class EncoreSolver():
    def __init__(self, model, instance, reference_instances, n_binary_variables, _hash=str(uuid.uuid4().fields[-1])[:8]):
        super().__init__()
        self._hash = _hash
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        
        self.model = model
        self.instance = (instance, )
        self.reference_instances = reference_instances
        self.n_binary_variables = n_binary_variables

        self.model_filename = "tmp" + os.sep + "encore-" + self._hash + "-model.cnf"
        self.instance_filename = "tmp" + os.sep + "encore-" + self._hash + "-instance.cnf"
        self.reference_instances_filenames = ["tmp" + os.sep + "encore-" + self._hash + "-reference-instances-" + str(label) +".cnf" for label in reference_instances.keys()]

        



    def to_cnf_file(self, filename, cnf, n_binary_variables):
        file = open(filename, "w")
        file.write(f"p cnf {n_binary_variables} {len(cnf)}\n")
        print("cnf:", cnf)
        for clause in cnf:
            file.write(" ".join(str(lit) for lit in clause) + " 0\n")
        file.close()





    def solve(self, *, n_anchors, time_limit=0):
        if time_limit != 0:
            raise NotImplementedError("TO DO IN ENCORE")
        
        self.to_cnf_file(self.model_filename, self.model, self.n_binary_variables)
        self.to_cnf_file(self.instance_filename, self.instance, self.n_binary_variables)
        
        if list(self.reference_instances.keys()) != [0, 1]:
            raise NotImplementedError("Only binary classification problem can be taken into account: "+str(list(self.reference_instances.keys())))

        for i, label in enumerate(self.reference_instances.keys()):
            filename = self.reference_instances_filenames[i]
            self.to_cnf_file(filename, self.reference_instances[label], self.n_binary_variables)
        
        command = [ENCORE_EXEC]        
        command += ["-i"]+[self.instance_filename]
        command += ["-n"]+[self.reference_instances_filenames[0]]
        command += ["-p"]+[self.reference_instances_filenames[1]]
        command += ["-x"]+[self.instance_filename]
        command += ["-k"]+[str(n_anchors)]
        command += ["-m toto"]
        print("command:", " ".join(command))
        
        time_used = -time.time()
        p = subprocess.run(command, timeout=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        time_used += time.time()
        print("ess:", p.stderr.split(os.linesep))
        output_str = [line.split(" ") for line in p.stdout.split(os.linesep)]
        print("output_str:", output_str)

        #./build/encore -i instances/paper_example/positive_f.cnf -p instances/paper_example/ac_moins.txt -n instances/paper_example/ac_plus.txt -x instances/paper_example/x.inst -k 2 -m toto

        return None
