from pyxai import Builder, Learning, Explainer, Tools
import unittest

Tools.set_verbose(0)


class TestBT(unittest.TestCase):
    model = None

    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Xgboost("tests/compas.csv", learner_type=Learning.CLASSIFICATION)
            cls.model = cls.learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT)
        return cls.learner, cls.model


    def test_tree_specific(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=30)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            tree_specific_reason = explainer.tree_specific_reason()
            self.assertTrue(explainer.is_tree_specific_reason(tree_specific_reason))


    def test_excluded(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        explainer.set_excluded_features(['African_American', 'Hispanic'])
        instances = learner.get_instances(model, n=30)

        for instance, prediction in instances:
            explainer.set_instance(instance)
            tree_specific_reason = explainer.tree_specific_reason()
            self.assertFalse(explainer.reason_contains_features(tree_specific_reason, 'Hispanic'))
            self.assertFalse(explainer.reason_contains_features(tree_specific_reason, 'African_American'))

        explainer.set_excluded_features(['Female'])
        for instance, prediction in instances:
            explainer.set_instance(instance)
            tree_specific_reason = explainer.tree_specific_reason()
            self.assertFalse(explainer.reason_contains_features(tree_specific_reason, 'Female'))

    def test_contrastive(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        instances = learner.get_instances(model, n=5)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            contrastive_reason = explainer.minimal_contrastive_reason()
            self.assertTrue(len(contrastive_reason) > 0 and explainer.is_contrastive_reason(contrastive_reason))

if __name__ == '__main__':
    unittest.main(verbosity=2)
