import c_explainer
from pyxai.sources.core.explainer.Explainer import Explainer
from pyxai.sources.core.explainer.explainerBT import ExplainerBT
from pyxai.sources.core.structure.type import ReasonExpressivity
from pyxai.sources.solvers.CPLEX.SufficientRegressionBT import SufficientRegression


class ExplainerRegressionBT(ExplainerBT):
    def __init__(self, boosted_trees, instance=None):
        self._lower_bound = None
        self._upper_bound = None
        super().__init__(boosted_trees, instance)


    def set_instance(self, instance):
        super().set_instance(instance)
        self._lower_bound = self.predict(instance)
        self._upper_bound = self._lower_bound


    @property
    def regression_boosted_trees(self):
        return self._boosted_trees


    def set_interval(self, lower_bound, upper_bound):
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
        return self._boosted_trees.predict_instance(instance)


    def predict_implicant(self, bin_implicant):
        return self._boosted_trees.predict_implicant(bin_implicant

                                                     )


    def tree_specific_reason(self, *, n_iterations=50, time_limit=None, seed=0):
        if self._instance is None:
            raise ValueError("Instance is not set")

        reason_expressivity = ReasonExpressivity.Conditions
        if self._upper_bound is None or self.lower_bound is None:
            raise RuntimeError("lower bound and upper bound must be set when computing a reason")
        if seed is None:
            seed = -1
        if time_limit is None:
            time_limit = 0

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
        result = c_explainer.compute_reason(self.c_BT, self._binary_representation, self._implicant_id_features, 0, n_iterations,
                                          time_limit,
                                          int(reason_expressivity), seed)
        self.add_history(self._instance, self.__class__.__name__, self.tree_specific_reason.__name__, result)
        return result

    def sufficient_reason(self, *, seed=0, time_limit=None):
        if self._instance is None:
            raise ValueError("Instance is not set")

        cplex = SufficientRegression()
        reason, time_used = cplex.create_model_and_solve(self, self._lower_bound, self._upper_bound)
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        self.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__, reason)
        return reason


    def extremum_range(self):
        min_weights = []
        max_weights = []
        for tree in self._boosted_trees.forest:
            leaves = tree.get_leaves()
            min_weights.append(min([l.value for l in leaves]))
            max_weights.append(max([l.value for l in leaves]))
        return (sum(min_weights), sum(max_weights))


    def is_implicant(self, abductive):
        min_weights = []
        max_weights = []
        base_score = self.regression_boosted_trees.learner_information.extras["base_score"]

        for tree in self._boosted_trees.forest:
            weights = self.compute_weights(tree, tree.root, abductive)
            min_weights.append(min(weights))
            max_weights.append(max(weights))
        return base_score + sum(min_weights) >= self._lower_bound and base_score + sum(max_weights) <= self._upper_bound
