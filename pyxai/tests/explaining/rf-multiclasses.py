from pyxai import Builder, Learning, Explaining, Tools
import unittest

Tools.set_verbose(0)


class TestExplainingRFMulticlasses(unittest.TestCase): # TODO merge with rf.py ?
    model = None


    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Scikitlearn("tests/datasets/iris.csv", problem_type='classification', instances_type='tabular', labels_type='classes')
            cls.model = cls.learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF)
        return cls.learner, cls.model


    def test_direct(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=5)
        for instance, _ in instances:
            explainer.set_instance(instance)
            direct_reason = explainer.direct_reason()

    def test_sufficient(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=5)
        for instance, _ in instances:
            explainer.set_instance(instance)
            sufficient_reason = explainer.sufficient_reason(time_limit=5)
            if explainer.elapsed_time == Explaining.TIMEOUT:
                print("Timeout => no tests to determine if reason is sufficient")
                self.assertTrue(explainer.is_reason(sufficient_reason))
            else:
                self.assertTrue(explainer.is_sufficient_reason(sufficient_reason) != False)


    def test_majoritary(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=5)
        for instance, _ in instances:
            explainer.set_instance(instance)
            majoritary_reason = explainer.majoritary_reason()
            self.assertTrue(explainer.is_majoritary_reason(majoritary_reason))


    def test_minimal_majoritary(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)  # ), features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=10)
        for instance, _ in instances:
            explainer.set_instance(instance)
            majoritary_reason = explainer.minimal_majoritary_reason(time_limit=5)
            self.assertTrue(len(majoritary_reason) == 0 or explainer.is_majoritary_reason(majoritary_reason))


    def test_contrastive(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=10)
        for instance, _ in instances:
            explainer.set_instance(instance)
            contrastive_reason = explainer.minimal_contrastive_reason(time_limit=5)
            self.assertTrue(len(contrastive_reason) == 0 or explainer.is_contrastive_reason(contrastive_reason))


if __name__ == '__main__':
    unittest.main(verbosity=2)
