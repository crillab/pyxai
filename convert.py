#python3 examples/Converters/converter-adult.py -dataset=examples/datasets_not_converted/adult.data

import sys
import os
import subprocess
import numpy
from statistics import median

assert len(sys.argv) == 2, "You have to put the dataset directory please !"
dataset = sys.argv[1]

for filename in os.listdir(dataset):
    completename = os.path.join(dataset, filename)
    
    if not completename.endswith("~"):
        print(completename)
        name = completename.split("/")[-1].split(".")[0].split("_")[0]
        print("name:", name)
        if name == "mnist49" or name == "mnist38":
            name = "mnist"
        command = "python3 examples/Converters/converter-"+name+".py -dataset=" + completename
        os.system(command)
        input("Press Enter to continue...")
