import sys
import os
import shutil
import pyxai

from pyxai.sources.core.tools.option import Options
from pyxai.sources.core.tools.utils import set_verbose
from pyxai.sources.core.tools.GUIQT import GraphicalInterface

__python_version__ = str(sys.version).split(os.linesep)[0].split(' ')[0]
__pyxai_version__ = open(os.path.join(os.path.dirname(__file__), 'version.txt'), encoding='utf-8').read()
__pyxai_location__ = os.sep.join(pyxai.__file__.split(os.sep)[:-1])

Options.set_values("dataset", "model_directory", "n_iterations", "time_limit", "verbose", "types", "output")
Options.set_flags("f", "gui", "examples", "explanations", "tests")
Options.parse(sys.argv[1:])
if Options.verbose is not None: set_verbose(Options.verbose)

if sys.argv:
    if  (len(sys.argv) != 0 and sys.argv[0] == "-m"):
        print("Python version: ", __python_version__)
        print("PyXAI version: ", __pyxai_version__)
        print("PyXAI location: ", __pyxai_location__)

    if  (len(sys.argv) == 1 and sys.argv[0] == "-m") or (len(sys.argv) == 2 and sys.argv[0] == "-m" and sys.argv[1] == "-gui"): 
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
        status = os.WEXITSTATUS(os.system("python3 tests/tests.py"))
        os.chdir(save_directory)
        sys.exit(status) 
    

                