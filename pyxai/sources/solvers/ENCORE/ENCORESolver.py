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
    def __init__(self, model, prediction, instance, reference_instances, n_binary_variables, _hash=str(uuid.uuid4().fields[-1])[:8]):
        super().__init__()
        self._hash = _hash
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        
        self.model = model
        self.prediction = prediction
        self.instance = (instance, )
        self.reference_instances = reference_instances
        self.n_binary_variables = n_binary_variables

        self.model_filename = "tmp" + os.sep + "encore-" + self._hash + "-model.cnf"
        self.instance_filename = "tmp" + os.sep + "encore-" + self._hash + "-instance.cnf"
        self.reference_instances_filenames = ["tmp" + os.sep + "encore-" + self._hash + "-reference-instances-" + str(label) +".cnf" for label in reference_instances.keys()]

        

    def to_instance_file(self, filename, cnf):
        file = open(filename, "w")
        for clause in cnf:
            file.write(" ".join(str(lit) for lit in clause))
        file.close()

    def to_cnf_file(self, filename, cnf, n_binary_variables):
        file = open(filename, "w")
        file.write(f"p cnf {n_binary_variables} {len(cnf)}\n")
        for clause in cnf:
            file.write(" ".join(str(lit) for lit in clause) + " 0\n")
        file.close()

    def solve(self, *, n_anchors, time_limit=None):
        
        self.to_cnf_file(self.model_filename, self.model, self.n_binary_variables)
        self.to_instance_file(self.instance_filename, self.instance)
        
        if list(self.reference_instances.keys()) != [0, 1]:
            raise NotImplementedError("Only binary classification problem can be taken into account: "+str(list(self.reference_instances.keys())))

        for i, label in enumerate(self.reference_instances.keys()):
            filename = self.reference_instances_filenames[i]
            self.to_cnf_file(filename, self.reference_instances[label], self.n_binary_variables)
        
        if self.prediction == 0:
             # When the instance is negative, we exchange expert knowledge. 
             self.reference_instances_filenames[0], self.reference_instances_filenames[1] = self.reference_instances_filenames[1], self.reference_instances_filenames[0]
        
        command = [ENCORE_EXEC]        
        command += ["-i"]+[self.model_filename]
        command += ["-n"]+[self.reference_instances_filenames[0]]
        command += ["-p"]+[self.reference_instances_filenames[1]]
        command += ["-x"]+[self.instance_filename]
        command += ["-k"]+[str(n_anchors)]
        command += ["-m toto"]
        #print("command:", " ".join(command))
        
        time_used = -time.time()
        try:
            p = subprocess.run(command, timeout=time_limit, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        except subprocess.TimeoutExpired:
            return "TIME_OUT", None, None
        
        time_used += time.time()
        output_str = [line.split(" ") for line in p.stdout.split(os.linesep)]
        #print("output_str:", output_str)

        status = [line.split(" ")[1] for line in p.stdout.split(os.linesep) if len(line) > 0 and line[0] == "s"][0]
        reason = [int(lit) if (lit != "v" and lit != "0") else "" for line in output_str for lit in line if line[0] == "v"]
        reason = tuple(lit for lit in reason if lit != "")
        return status, reason, time_used
    
        
