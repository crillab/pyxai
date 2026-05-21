"""Through to the ``initialize`` function, this module allows to create an explainer object dedicated to finding explanations from a ``model`` and an instance ``instance``.

The produced explainer object contains the following useful methods: 
  - The ``set_instance`` method allows to change the instance under consideration. 

  - Several types of reasons can be calculated according to the type of model (decision tree, random forest, boosted tree) using methods: ``direct_reason``, ``contrastive_reason``, ``sufficient_reason``, ``minimal_sufficient_reason``, ``preferred_reason``, .... All produced reasons are in binary form where each binary variable represents a condition (feature < value ?) in the decision tree. 

  - The ``to_features`` method allows to convert a reason in binary form into a form composed of features.   

  - The ``is_implicant``, ``is_sufficient`` and ``is_contrastive`` methods allow to check some key properties on the reasons.

Other model-specific functions are also available. Please see the docstring of ExplainerDT, ExplainerRF or ExplainerBT for more details. 
"""
from typing import TypeVar

from pyxai.sources.explainers.Explainer import Explainer 
from pyxai.sources.explainers.explainerBT import ExplainerBT
from pyxai.sources.explainers.explainerDT import ExplainerDT
from pyxai.sources.explainers.explainerRF import ExplainerRF
from pyxai.sources.explainers.explainerRegressionBT import ExplainerRegressionBT
from pyxai.sources.explainers.explainerLinear import ExplainerLinear

from pyxai.sources.data_structures.boostedTrees import BoostedTrees
from pyxai.sources.data_structures.boostedTreesRegression import BoostedTreesRegression

from pyxai.sources.data_structures.decisionTree import DecisionTree
from pyxai.sources.data_structures.randomForest import RandomForest
from pyxai.sources.data_structures.linearModel import LinearModel



from pyxai.sources.type import TypeReason, TypeStatus, ReasonExpressivity, PreferredReasonMethod, TypeTheory

def show():
    from pyxai.sources.core.tools.GUIQT import GraphicalInterface
    graphical_interface = GraphicalInterface(None)
    graphical_interface.mainloop()

def decision_tree(model, instance=None):
    return ExplainerDT(model, instance)


def random_forest(model, instance=None):
    return ExplainerDT(model, instance)


def boosted_trees(model, instance=None):
    return ExplainerBT(model, instance)


def initialize(model, instance=None, features_type=None):
    """Return and initialize an explainer according to a model and optionally an instance and a domain theorie.
       
    Parameters
    ----------

    model: ```DecisionTree``` | ```RandomForest``` | ```BoostedTrees``` | ```BoostedTreesRegression```
        A PyXAI model. 
        
    instance: ```list[int]``` (optional, default=```None```)
        The instance (an observation) on which explanations must be calculated.

    features_type: ```dict``` | ```str``` (optional, default=```None```)
        Either a Python dictionnary containing the keys `numerical`, `categorical` and `binary` or the file path of a .types file. 
        Activate a domain theorie. 

    Returns
    -------
    ```ExplainerDT``` | ```ExplainerRF``` | ```ExplainerBT``` | ```ExplainerRegressionBT```:
       The appropriate explainer according to ```model```.
    """
    
        
    explainer = None
    if isinstance(model, DecisionTree):
        explainer = ExplainerDT(model, instance)
    elif isinstance(model, RandomForest):
        explainer = ExplainerRF(model, instance)
    elif isinstance(model, BoostedTrees):
        if isinstance(model, BoostedTreesRegression):
            explainer = ExplainerRegressionBT(model, instance)
        else:
            explainer = ExplainerBT(model, instance)

    elif isinstance(model, LinearModel):
        explainer = ExplainerLinear(model, instance)
    else: 
        raise ValueError("The given model is not supported: "+str(type(model)))
    
    if features_type is not None:
        explainer.set_features_type(features_type)

    return explainer


UNSAT = TypeStatus.Unsat
""" Solver status: unsatisfiable (means that no solution is found by the solver) """

SAT = TypeStatus.Sat
""" Solver status: satisfiable (means that at least one solution is found by the solver) """

OPTIMUM = TypeStatus.Optimum
""" Solver status: optimum (means that an optimal solution is found by the solver) """

CORE = TypeStatus.Core
""" Solver status: core (means that an unsatisfiable core has been extracted by the solver) """

UNKNOWN = TypeStatus.Unknown
""" Solver status: unknown (means that the solver is unable to solve the problem instance)  """

ALL = TypeReason.All

FEATURES = ReasonExpressivity.Features
CONDITIONS = ReasonExpressivity.Conditions

MINIMAL = PreferredReasonMethod.Minimal
FEATURE_IMPORTANCE = PreferredReasonMethod.FeatureImportance
SHAPLEY = PreferredReasonMethod.Shapley
WEIGHTS = PreferredReasonMethod.Weights
WORD_FREQUENCY = PreferredReasonMethod.WordFrequency
WORD_FREQUENCY_LAYERS = PreferredReasonMethod.WordFrequencyLayers
INCLUSION_PREFERRED = PreferredReasonMethod.InclusionPreferred

TIMEOUT = Explainer.TIMEOUT

