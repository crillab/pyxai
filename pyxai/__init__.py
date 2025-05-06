import sys
import os
import shutil
import pyxai
import platform
import subprocess
import unittest
import matplotlib

__version__ = "1.0.15"

matplotlib.set_loglevel("critical") #To win a lot of times. 

from pyxai.sources.core.tools.option import Options
from pyxai.sources.core.tools.utils import set_verbose, check_PyQt6

__python_version__ = str(sys.version).split(os.linesep)[0].split(' ')[0]
__pyxai_location__ = os.sep.join(pyxai.__file__.split(os.sep)[:-1])

Options.set_values("dataset", "model_directory", "n_iterations", "time_limit", "verbose", "types", "output", "type_references", "N")
Options.set_flags("f", "gui", "examples", "explanations", "tests")
Options.parse(sys.argv[1:])
if Options.verbose is not None: set_verbose(Options.verbose)

if sys.argv:
        if  (len(sys.argv) != 0 and sys.argv[0] == "-m"):
                print("Python version: ", __python_version__)
                print("PyXAI version: ", __version__)
                print("PyXAI location: ", __pyxai_location__)

        if  (len(sys.argv) == 1 and sys.argv[0] == "-m") or (len(sys.argv) == 2 and sys.argv[0] == "-m" and sys.argv[1] == "-gui"):         
            check_PyQt6()
            
            from pyxai.sources.core.tools.GUIQT import GraphicalInterface
            graphical_interface = GraphicalInterface(None)
            graphical_interface.mainloop()
        elif (len(sys.argv) == 2 and sys.argv[0] == "-m" and sys.argv[1] == "-examples"):
            
            
            examples = __pyxai_location__ + os.sep + "examples" + os.sep
            target = os.getcwd() + os.sep + "examples" + os.sep
            print("Source of files found: ", examples)
            shutil.copytree(examples, target, ignore=shutil.ignore_patterns('in_progress', '__init__.py', '__pycache__*'))
            print("Successful creation of the " + target + " directory containing the examples.")
            exit(0)

        elif (len(sys.argv) == 2 and sys.argv[0] == "-m" and sys.argv[1] == "-explanations"):
            
            explanations = __pyxai_location__ + os.sep + "explanations" + os.sep
            target = os.getcwd() + os.sep + "explanations" + os.sep
            print("Source of files found: ", explanations)
            shutil.copytree(explanations, target, ignore=shutil.ignore_patterns('in_progress', '__init__.py', '__pycache__*'))
            print("Successful creation of the " + target + " directory containing the explanations.")
            exit(0)
        
        elif (len(sys.argv) == 2 and sys.argv[0] == "-m" and sys.argv[1] == "-tests"):
            save_directory = os.getcwd()
            os.chdir(__pyxai_location__)
            print("Change directory to PyXAI location: ", __pyxai_location__)
            cmd = "python3 -u tests"+os.sep+"tests.py -f"
            cmd = cmd.split(" ")
            print("command:", cmd)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, errors = process.communicate()
            #stdout = process.stdout.read().decode('utf-8')
            #stderr = process.stderr.read().decode('utf-8')
            print("stdout:", output.decode('utf-8'))
            print("stderr:", errors.decode('utf-8'))

            #print("stderr:", stderr)
            #print("return code:", process.returncode)
            #print("return code2:", errors)
            os.chdir(save_directory)
            exit(process.poll())
            #if platform.system() == "Windows":
            #    exit(status)
            #else:
            #    exit(os.WEXITSTATUS(status)) 
            

   
        

                
