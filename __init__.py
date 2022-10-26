import sys

from pyxai.sources.core.tools.option import Options
from pyxai.sources.core.tools.utils import set_verbose

# from pyxai.solvers.ML.scikitlearn import Scikitlearn
# from pyxai.solvers.ML.xgboost import Xgboost

# from pyxai.core.tools.utils import display_observation, Stopwatch

# from pyxai.core.explainer.explainerDT import ExplainerDT
# from pyxai.core.explainer.explainerBT import ExplainerBT
# from pyxai.core.explainer.explainerRF import ExplainerRF

# from pyxai.core.structure.type import TypeReason, TypeCount, TypeTree, EvaluationMethod, EvaluationOutput, Indexes, ReasonExpressivity
# from pyxai.core.tools.heatmap import HeatMap
# from pyxai.core.structure.decisionTree import DecisionTree, DecisionNode
# from pyxai.core.structure.randomForest import RandomForest

# from pyxai.core.structure.boostedTrees import BoostedTrees


# DIRECT = TypeReason.Direct
# SUFFICIENT = TypeReason.Sufficient
# MINIMAL_SUFFICIENT = TypeReason.MinimalSufficient
# ALL = TypeReason.All
Options.set_values("dataset", "model_directory", "n_iterations", "time_limit", "verbose")
Options.set_flags("f")
Options.parse(sys.argv[1:])
if Options.verbose is not None: set_verbose(Options.verbose)
