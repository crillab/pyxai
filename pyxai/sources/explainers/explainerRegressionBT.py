import c_explainer
from pyxai.sources.explainers.Explainer import Explainer
from pyxai.sources.explainers.explainerBT import ExplainerBT
from pyxai.sources.type import ReasonExpressivity
from pyxai.sources.solvers.MIP.SufficientRegressionBT import SufficientRegression
from pyxai.sources.solvers.MIP.Range import Range
import time


### 
# TODO: missing contrastive explanations for regression boosted trees
# https://gitlab.univ-artois.fr/expekctation/software/btex
###

class ExplainerRegressionBT(ExplainerBT):
    r"""The Boosted Trees Regression explainer.

    This class adapts the generic ```Explainer``` to a
    ```BoostedTreesRegression``` model.  It provides methods to:

    - set an instance and derive its Boolean (literal-based) representation,
    - compute different families of explanations (direct, contrastive, tree_specific),

    Some functions come from the parent class ```Explainer```, their documentation is available in the dedicated page:
    - Main Methods: ```set_instance```, ```set_features_type```, ```get_model```, ```predict```, ```activate_theory```, ```deactivate_theory```, ```set_excluded_features```, ```unset_excluded_features```;
    - Human-Readable Explanation Methods: ```to_features```, ```get_feature_names```, ```get_feature_names_from_literal```, ```reason_contains_features```;
    - Explanation verification methods: ```is_implicant```, ```is_reason```, ```is_sufficient_reason```, ```is_contrastive_reason```.

    Parameters
    ----------
    lower_bound: ```float```
        The lower bound used for explanations.
    upper_bound: ```float```
        The upper bound used for explanations.


    Raises
    ------
    ValueError:
        If the instance is not set.
    RuntimeError:
        If the lower or upper bound are not set.

    Notes
    -----
    Except for the direct reason, one has to provide an interval, for which the prediction is acceptable for the user.

    See Also
    --------
    - The [documentation pages](https://www.cril.univ-artois.fr/pyxai/documentation/regression/)
    - The paper [Computing Abductive Explanations for Boosted Regression Trees ](https://www.ijcai.org/proceedings/2023/382)
    """

    def __init__(self, boosted_trees, instance=None):
        """
        Initialize an explainer for a Boosted Tree model.

        Parameters
        ----------
        boosted_trees: ```BoostedTreesRegression```
            The BoostedTrees model to explain.
        instance : ```list[int]``` (optional, default: None)
            The instance  for which explanations will be computed.
            If ``None``, you must call set_instance before requesting explanations.

        """
        self._lower_bound = None
        self._upper_bound = None
        super().__init__(boosted_trees, instance)


    def set_instance(self, instance):
        r"""Set the instance to explain.

         Once the Explainer is built, you can explain several instances using it.
         Each time you want to explain a new instance, you need to call the function set_instance
         (it is not necessary to create a new Explainer object).

         Parameters
         ----------
         instance : ```list[float]```
             The instance to explain.
         """
        super().set_instance(instance)
        self._lower_bound = self.predict(instance)
        self._upper_bound = self._lower_bound


    @property
    def regression_boosted_trees(self):
        return self._boosted_trees


    def set_interval(self, lower_bound, upper_bound):
        """Set the interval.

        For the regression case, one has to provide an interval, for which the prediction is acceptable for the user.
        Set the interval for the reason. The larger the interval is, the smaller the reason is.

        Parameters
        ----------
        lower_bound: ```float```
            The lower bound of the interval
        upper_bound: ```float```
            The upper bound of the interval
        """
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound


    @property
    def lower_bound(self):
        return self._lower_bound


    @lower_bound.setter
    def lower_bound(self, lower_bound):
        self._lower_bound = lower_bound


    @property
    def upper_bound(self):
        return self._upper_bound


    @upper_bound.setter
    def upper_bound(self, upper_bound):
        self._upper_bound = upper_bound


    def predict(self, instance):
        r"""Predict the label/value for an instance using the underlying tree.

        Parameters
        ----------
        instance: ```list``` of ```float```
            Observation to classify/regress.

        Returns
        -------
        ```float```:
            The prediction for the given instance.
        """
        return self._boosted_trees.predict_instance(instance)


    def predict_implicant(self, bin_implicant):
        return self._boosted_trees.predict_implicant(bin_implicant)


    def tree_specific_reason(self, *, n_iterations=50, time_limit=None, seed=0):
        r"""Compute a tree specific reason for the current instance.

        Informally, a tree specific  reason for classifying is a prime implicant t of a majority of decision trees in f that covers
        the instance.

        Parameters
        ----------
        n_iterations: ```int``` (optional, default: ```50```)
             Number of randomized iterations used to extract the smallest one.
        time_limit: ```float``` (optional,  default=```None```)
            Time limit (seconds) for the whole enumeration. If reached, `elapsed_time` is set to ```Explaining.TIMEOUT```.
        seed: ```int``` (optional, default: ```0```)
            A seed to change the reason. 
            
        Returns
        -------
        ```list[int]```:
            The tree specific reason.
        ```None```:
            If computed reasons contain excluded features.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/regression/BTregression/treespecific/)

        Raises
        ------
        ValueError
            If no instance has been set.
        RuntimeError
            If lower or upper bound are not specified.
        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        if self._upper_bound is None or self.lower_bound is None:
            raise RuntimeError("lower bound and upper bound must be set when computing a reason")
        if seed is None:
            seed = -1
        if time_limit is None:
            time_limit = 0
        reason_expressivity = ReasonExpressivity.Conditions
        if self.c_BT is None:
            # Preprocessing to give all trees in the c++ library
            self.c_BT = c_explainer.new_regression_BT()
            for tree in self._boosted_trees.forest:
                c_explainer.add_tree(self.c_BT, tree.raw_data_for_CPP())
            c_explainer.set_base_score(self.c_BT, self._boosted_trees.learner_information.extras["base_score"])
        c_explainer.set_excluded(self.c_BT, tuple(self._excluded_literals))
        if self._theory:
            c_explainer.set_theory(self.c_BT, tuple(self._boosted_trees.get_theory(self._binary_representation)))
        c_explainer.set_interval(self.c_BT, self._lower_bound, self._upper_bound)
        # 0 for prediction. We don't care of it. The interval is the important thing here
        result = c_explainer.compute_reason(self.c_BT, self._binary_representation, self._implicant_id_features, 
                                            0, n_iterations,
                                            time_limit,
                                            reason_expressivity, seed, 0)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.tree_specific_reason.__name__, result)
        return result

    def sufficient_reason(self, *, seed=0, time_limit=None):
        raise NotImplemented("sufficient reason is not yet implemented for regression boosted trees")
        if self._instance is None:
            raise ValueError("Instance is not set")

        cplex = SufficientRegression()
        reason, time_used = cplex.create_model_and_solve(self, self._lower_bound, self._upper_bound)
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__, reason)
        return reason


    def extremum_range(self):
        r"""The extremum range for predictions for the model.

        The extremum range is computed in polynomial time, but the real extremum range can be smaller.
        Use range_for_partial_instance(self, [None for _ in range(explainer.instance]) to extract the real one (can be time consuming).


        Returns
        -------
        ```(float,float)```:
            A tuple (min_value, max_value) corresponding to the extremum range.

        """
        min_weights = []
        max_weights = []
        for tree in self._boosted_trees.forest:
            leaves = tree.get_leaves()
            min_weights.append(min([l.value for l in leaves]))
            max_weights.append(max([l.value for l in leaves]))
        return (sum(min_weights), sum(max_weights))



    def range_for_partial_instance(self, partial_instance, *, time_limit=None):
        r"""Extract the range for all possible extension of the partial instance.

            Given a partial instance, extract the range for all possible extension of the partial instance.
            The partial instance is defined with None value on undefined values.
        
        Parameters
        ----------
        partial_instance: ```list[float]``` (optional, default: ```50```)
            An instance. 
        time_limit: ```float``` (optional,  default=```None```)
            Time limit (seconds) for the whole enumeration. If reached, `elapsed_time` is set to ```Explaining.TIMEOUT```.
        
        Note
        ----
        This function can be time consuming.

        Returns
        -------
        ```(float,float)```:
            A tuple (min_value, max_value) corresponding to the extremum range.
            """
        starting_time = -time.process_time()
        range = Range()
        partial = self._boosted_trees.instance_to_binaries(partial_instance)
        min_prediction = range.create_model_and_solve(self, None if self._theory is False else self._theory_clauses(), partial, True, time_limit)
        max_prediction = range.create_model_and_solve(self, None if self._theory is False else self._theory_clauses(), partial, False, time_limit)
        time_used = starting_time + time.process_time()
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        base_score = self._boosted_trees.learner_information.extras["base_score"]
        return [None, None] if self._elapsed_time == Explainer.TIMEOUT else [min_prediction + base_score, max_prediction+base_score]


    def __is_tree_specific_reason(self, abductive):
        r"""Check if a partial binary representation is an implicant (according to the acceptable range).

        Return
        ------
        ```bool```
            True if it is an implicant, False otherwise.

        """
        if self._boosted_trees.learner_information.problem_type != 'regression':
            raise NotImplementedError("The method is_tree_specific_reason is only implemented for regression boosted trees.")
        
        min_weights = []
        max_weights = []
        base_score = self.regression_boosted_trees.learner_information.extras["base_score"]

        for tree in self._boosted_trees.forest:
            weights = self.compute_weights(tree, tree._root, abductive)
            min_weights.append(min(weights))
            max_weights.append(max(weights))
        return base_score + sum(min_weights) >= self._lower_bound and base_score + sum(max_weights) <= self._upper_bound
