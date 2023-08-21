from pyxai import Builder, Learning, Explainer, Tools
import unittest

Tools.set_verbose(0)


class TestDT(unittest.TestCase):
    model = None


    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Scikitlearn("tests/compas.csv", learner_type=Learning.CLASSIFICATION)
            cls.model = cls.learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
        return cls.learner, cls.model


    def test_sufficients(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        instances = learner.get_instances(model, n=30)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            sufficient_reasons = explainer.sufficient_reason(n=10)
            for sr in sufficient_reasons:
                self.assertTrue(explainer.is_sufficient_reason(sr))


    def test_contrastives(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        instances = learner.get_instances(model, n=30)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            contrastives_reasons = explainer.contrastive_reason(n=10)
            for sr in contrastives_reasons:
                self.assertTrue(explainer.is_contrastive_reason(sr))


    def test_minimals(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        instances = learner.get_instances(model, n=30)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            minimal_reasons = explainer.minimal_sufficient_reason(n=10)
            for m in minimal_reasons:
                self.assertTrue(explainer.is_sufficient_reason(m))


    def test_excluded(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        explainer.set_excluded_features(['African_American', 'Hispanic'])
        instances = learner.get_instances(model, n=30)

        for instance, prediction in instances:
            explainer.set_instance(instance)
            sufficient_reasons = explainer.sufficient_reason(n=10)
            for sr in sufficient_reasons:
                self.assertFalse(explainer.reason_contains_features(sr, 'Hispanic'))
                self.assertFalse(explainer.reason_contains_features(sr, 'African_American'))

            contrastives_reasons = explainer.contrastive_reason(n=10)
            for sr in contrastives_reasons:
                self.assertFalse(explainer.reason_contains_features(sr, 'Hispanic'))
                self.assertFalse(explainer.reason_contains_features(sr, 'African_American'))

        explainer.set_excluded_features(['Female'])
        for instance, prediction in instances:
            explainer.set_instance(instance)
            sufficient_reasons = explainer.sufficient_reason(n=10)
            for sr in sufficient_reasons:
                self.assertFalse(explainer.reason_contains_features(sr, 'Female'))

            contrastives_reasons = explainer.contrastive_reason(n=10)
            for sr in contrastives_reasons:
                self.assertFalse(explainer.reason_contains_features(sr, 'Female'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
