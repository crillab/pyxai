from pyxai import Builder, Learning, Explainer, Tools
import unittest

Tools.set_verbose(0)


class TestDT(unittest.TestCase):
    model = None


    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Xgboost("tests/iris.csv", learner_type=Learning.CLASSIFICATION)
            cls.model = cls.learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT)
        return cls.learner, cls.model


    def test_tree_specific(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=30)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            tree_specific_reason = explainer.tree_specific_reason()
            self.assertTrue(explainer.is_tree_specific_reason(sufficient_reason))


if __name__ == '__main__':
    unittest.main(verbosity=2)
