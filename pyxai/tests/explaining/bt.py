from pyxai import Builder, Learning, Explaining, Tools
import unittest

Tools.set_verbose(0)


class TestExplainingBT(unittest.TestCase):
    model = None

    def test_implicant_and_tree_specific(self):
        learner = Learning.Xgboost("tests/datasets/12_fault.csv", problem_type='classification')
        self.main_implicant_and_tree_specific(learner)

    @unittest.skip("minimal_tree_specific_reasons not available in V2, to be re-introduced in V3")
    def test_tree_specific_numerical(self):
        learner = Learning.Xgboost("tests/datasets/12_fault.csv", problem_type='classification')
        self.main_minimal_tree_specifics(learner)

    def test_excluded(self):
        learner = Learning.Xgboost("tests/datasets/12_fault.csv", problem_type='classification')
        self.main_excluded(learner)

    @unittest.skip("contrastive reason is not yet implemented for regression boosted trees")
    def test_contrastive(self):
        learner = Learning.Xgboost("tests/datasets/12_fault.csv", problem_type='classification')
        self.main_contrastive(learner)


    @unittest.skip("sufficient reason is not yet implemented for regression boosted trees")
    def test_sufficient(self):
        learner = Learning.Xgboost("tests/datasets/winequality-red.csv", problem_type='regression', instances_type='tabular', labels_type='discrete-values')
        self.main_sufficient(learner)

    def main_implicant_and_tree_specific(self, learner):
        model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT, model_parameters={}, splitting_parameters={})
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=3)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            tree_specific_reason = explainer.tree_specific_reason()
            self.assertTrue(explainer.is_tree_specific_reason(tree_specific_reason))
            self.assertTrue(explainer.is_implicant(tree_specific_reason))

    def main_minimal_tree_specifics(self, learner):
        model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT, model_parameters={}, splitting_parameters={})
        explainer = Explaining.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=3)
        for instance, _ in instances:
            explainer.set_instance(instance)
            reason = explainer.minimal_tree_specific_reasons(n=1)
            self.assertTrue(explainer.is_tree_specific_reason(reason))
            self.assertTrue(explainer.is_implicant(reason))
            
    def main_excluded(self, learner):
        model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT, model_parameters={}, splitting_parameters={})
        explainer = Explaining.initialize(model)
        explainer.set_excluded_features(['African_American', 'Hispanic'])
        instances = learner.get_instances(model, n=3)
        for instance, _ in instances:
            explainer.set_instance(instance)
            tree_specific_reason = explainer.tree_specific_reason()
            self.assertFalse(explainer.reason_contains_features(tree_specific_reason, 'Hispanic'))
            self.assertFalse(explainer.reason_contains_features(tree_specific_reason, 'African_American'))
        explainer.set_excluded_features(['Female'])
        for instance, _ in instances:
            explainer.set_instance(instance)
            tree_specific_reason = explainer.tree_specific_reason()
            self.assertFalse(explainer.reason_contains_features(tree_specific_reason, 'Female'))

    def main_contrastive(self, learner):
        model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT, model_parameters={}, splitting_parameters={})
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=10)
        for instance, _ in instances:
            explainer.set_instance(instance)
            contrastive_reason = explainer.minimal_contrastive_reason()
            self.assertTrue(len(contrastive_reason) > 0 and explainer.is_contrastive_reason(contrastive_reason))

    def main_sufficient(self, learner):
        model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT, model_parameters={}, splitting_parameters={})
        explainer = Explaining.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=10)
        for instance, _ in instances:
            explainer.set_instance(instance)
            sufficient_reason = explainer.sufficient_reason(time_limit=5)
            self.assertTrue(explainer.is_reason(sufficient_reason))




if __name__ == '__main__':
    unittest.main(verbosity=2)
