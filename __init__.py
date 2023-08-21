import sys

from pyxai.sources.core.tools.option import Options
from pyxai.sources.core.tools.utils import set_verbose
from pyxai.sources.core.tools.GUIQT import GraphicalInterface

Options.set_values("dataset", "model_directory", "n_iterations", "time_limit", "verbose", "types", "output")
Options.set_flags("f")
Options.parse(sys.argv[1:])
if Options.verbose is not None: set_verbose(Options.verbose)

if sys.argv:
    if len(sys.argv) == 1 and sys.argv[0] == "-m":  # copy of models
        graphical_interface = GraphicalInterface(None)
        graphical_interface.mainloop()