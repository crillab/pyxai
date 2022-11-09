"""Through to the ``initialize`` function, this module allows to create an explainer object dedicated to finding explanations from a ``model`` and an instance ``instance``.

The produced explainer object contains the following useful methods: 
  - The ``set_instance`` method allows to change the instance under consideration. 

  - Several types of reasons can be calculated according to the type of model (decision tree, random forest, boosted tree) using methods: ``direct_reason``, ``contrastive_reason``, ``sufficient_reason``, ``minimal_sufficient_reason``, ``preferred_reason``, .... All produced reasons are in binary form where each binary variable represents a condition (feature < value ?) in the decision tree. 

  - The ``to_features`` method allows to convert a reason in binary form into a form composed of features.   

  - The ``is_implicant``, ``is_sufficient`` and ``is_contrastive`` methods allow to check some key properties on the reasons.

Other model-specific functions are also available. Please see the docstring of ExplainerDT, ExplainerRF or ExplainerBT for more details. 
"""
from typing import TypeVar

from pyxai.sources.core.explainer.Explainer import Explainer
from pyxai.sources.core.explainer.explainerBT import ExplainerBT
from pyxai.sources.core.explainer.explainerDT import ExplainerDT
from pyxai.sources.core.explainer.explainerRF import ExplainerRF
from pyxai.sources.core.structure.boostedTrees import BoostedTrees
from pyxai.sources.core.structure.decisionTree import DecisionTree
from pyxai.sources.core.structure.randomForest import RandomForest
from pyxai.sources.core.structure.type import TypeReason, TypeStatus, ReasonExpressivity, PreferredReasonMethod


def decision_tree(model, instance=None):
    return ExplainerDT(model, instance)


def random_forest(model, instance=None):
    return ExplainerDT(model, instance)


def boosted_trees(model, instance=None):
    return ExplainerBT(model, instance)


def initialize(model, instance=None):
    """Return and initialize an explainer according to a model and optionally an instance.

    Args:
        model (BoostedTrees, RandomForest, DecisionTree): A model.
        instance (:obj:`list` of :obj:`int`, optional): The instance (an observation) on which explanations must be calculated.. Defaults to None.

    Returns:
        ExplainerDT|ExplainerRF|ExplainerBT: The explainer according to ``model``.
    """
    if isinstance(model, DecisionTree):
        return ExplainerDT(model, instance)
    if isinstance(model, RandomForest):
        return ExplainerRF(model, instance)
    if isinstance(model, BoostedTrees):
        return ExplainerBT(model, instance)
    return None


UNSAT = TypeStatus.UNSAT
""" Solver status: unsatisfiable (means that no solution is found by the solver) """

SAT = TypeStatus.SAT
""" Solver status: satisfiable (means that at least one solution is found by the solver) """

OPTIMUM = TypeStatus.OPTIMUM
""" Solver status: optimum (means that an optimal solution is found by the solver) """

CORE = TypeStatus.CORE
""" Solver status: core (means that an unsatisfiable core has been extracted by the solver) """

UNKNOWN = TypeStatus.UNKNOWN
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
