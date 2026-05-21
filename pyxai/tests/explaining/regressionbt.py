from pyxai import Learning, Explaining, Tools
import unittest
import math

Tools.set_verbose(0)


class TestExplainingRegressionBT(unittest.TestCase):
    model = None

    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Xgboost(
                "tests/datasets/winequality-red.csv",
                problem_type='regression',
                instances_type='tabular',
                labels_type='discrete-values'
            )
            cls.model = cls.learner.evaluate(
                splitting_method=Learning.HOLD_OUT,
                model_type=Learning.BT,
                model_parameters={},
                splitting_parameters={}
            )
        return cls.learner, cls.model

    def test_tree_specific_reason(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=3)
        for instance, _ in instances:
            explainer.set_instance(instance)
            tree_specific_reason = explainer.tree_specific_reason()
            self.assertTrue(len(tree_specific_reason) > 0)

    def test_tree_specific_reason_with_interval(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=2)
        for instance, _ in instances:
            explainer.set_instance(instance)
            prediction = explainer.predict(instance)
            # Wider interval => reason can be smaller or equal
            explainer.set_interval(prediction - 0.5, prediction + 0.5)
            wide_reason = explainer.tree_specific_reason()
            self.assertTrue(len(wide_reason) > 0)

    def test_extremum_range(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        min_val, max_val = explainer.extremum_range()
        self.assertTrue(min_val <= max_val)
        # All predictions must fall within the extremum range
        instances = learner.get_instances(model, n=3)
        for instance, _ in instances:
            prediction = explainer.predict(instance)
            self.assertTrue(min_val <= prediction <= max_val)

    def test_range_for_partial_instance(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=3)
        for instance, _ in instances:
            explainer.set_instance(instance)
            # Partial instance: first feature unknown
            partial_instance = [None] + list(instance[1:])
            result = explainer.range_for_partial_instance(partial_instance, time_limit=5)
            if result != [None, None]:  # [None, None] means timeout
                min_val, max_val = result
                self.assertTrue(min_val <= max_val)
                prediction = explainer.predict(instance)
                self.assertTrue(math.floor(min_val) <= prediction <= math.ceil(max_val))

    def test_excluded(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        excluded_feature = 'alcohol'
        explainer.set_excluded_features([excluded_feature])
        instances = learner.get_instances(model, n=3)
        for instance, _ in instances:
            explainer.set_instance(instance)
            tree_specific_reason = explainer.tree_specific_reason()
            if tree_specific_reason is not None:
                self.assertFalse(explainer.reason_contains_features(tree_specific_reason, excluded_feature))

    @unittest.skip("sufficient reason is not yet implemented for regression boosted trees")
    def test_sufficient_reason(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=5)
        for instance, _ in instances:
            explainer.set_instance(instance)
            sufficient_reason = explainer.sufficient_reason()
            self.assertTrue(explainer.is_reason(sufficient_reason))


if __name__ == '__main__':
    unittest.main(verbosity=2)
