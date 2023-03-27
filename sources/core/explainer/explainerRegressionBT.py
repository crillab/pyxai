import c_explainer
from pyxai.sources.core.explainer.Explainer import Explainer
from pyxai.sources.core.explainer.explainerBT import ExplainerBT
from pyxai.sources.core.structure.type import ReasonExpressivity

class ExplainerRegressionBT(ExplainerBT) :
    def __init__(self, boosted_trees, instance=None):
        super().__init__(boosted_trees, instance)
        self._lower_bound = None
        self._upper_bound = None

    def set_range(self, lower_bound, upper_bound):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @property
    def lower_bound(self) :
        return self._lower_bound


    @lower_bound.setter
    def lower_bound(self, lower_bound):
        self._lower_bound = lower_bound

    @property
    def upper_bound(self) :
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, upper_bound):
        self._upper_bound = upper_bound

    def predict(self, instance) :
        return self._boosted_trees.predict_instance(instance)


    def extremum_range(self) :
        lb = float('inf')
        ub = float('-inf')
        for tree in self._boosted_trees:
            leaves = tree.get_leaves()

        return (lb, ub)


    def tree_specific_reason(self, *, n_iterations=50, time_limit=None, seed=0):
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
        return c_explainer.compute_reason(self.c_BT, self._binary_representation, self._implicant_id_features, 0, n_iterations,
                                            time_limit,
                                            int(reason_expressivity), seed)



    def extremum_range(self):
        minimum = []
        maximum = []
        for tree in self._boosted_trees.forest:
            leaves = tree.get_leaves()
            minimum.append(min([l.value for l in leaves]))
            maximum.append(max([l.value for l in leaves]))
        return (sum(minimum), sum(maximum))
