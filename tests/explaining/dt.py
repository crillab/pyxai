from pyxai import Builder, Learning, Explainer, Tools
import unittest

Tools.set_verbose(0)


class TestDT(unittest.TestCase):
    model = None


    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Scikitlearn("tests/iris.csv", learner_type=Learning.CLASSIFICATION)
            cls.model = cls.learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT, test_size=0.2, max_depth=6)
        return cls.learner, cls.model


    def setUp(self):
        print("..|In method:", self._testMethodName)


    def test_sufficient(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        instances = learner.get_instances(model, n=30, correct=True)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            sufficient_reasons = explainer.sufficient_reason(n=10)
            for sr in sufficient_reasons:
                self.assertTrue(explainer.is_sufficient_reason(sr))


    def test_contrastives(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        instances = learner.get_instances(model, n=30, correct=True)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            contrastives_reasons = explainer.contrastive_reason(n=10)
            for sr in contrastives_reasons:
                self.assertTrue(explainer.is_contrastive_reason(sr))


    def test_minimals(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        instances = learner.get_instances(model, n=30, correct=True)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            minimal_reasons = explainer.minimal_sufficient_reason(n=10)
            print("ici ", minimal_reasons)
            for m in minimal_reasons:
                self.assertTrue(explainer.is_sufficient_reason(m))
