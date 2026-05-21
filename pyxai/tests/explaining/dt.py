from pyxai import Builder, Learning, Explaining, Tools
import unittest

Tools.set_verbose(0)


class TestExplainingDT(unittest.TestCase):
    model = None


    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Scikitlearn("tests/datasets/compas.csv", problem_type='classification', instances_type='tabular')
            cls.model = cls.learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.DT)
        return cls.learner, cls.model


    def test_sufficients(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=5)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            sufficient_reasons = explainer.sufficient_reason(n=3)
            for sr in sufficient_reasons:
                self.assertTrue(explainer.is_sufficient_reason(sr))
            
    def test_contrastives(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=5)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            contrastives_reasons = explainer.contrastive_reason(n=3)
            for sr in contrastives_reasons:
                self.assertTrue(explainer.is_contrastive_reason(sr))


    def test_minimals(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=5)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            minimal_reasons = explainer.minimal_sufficient_reason(n=3)
            for m in minimal_reasons:
                self.assertTrue(explainer.is_sufficient_reason(m))


    def test_excluded(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        explainer.set_excluded_features(['Origin_African_American', 'Origin_Hispanic'])
        instances = learner.get_instances(model, n=5)

        for instance, prediction in instances:
            explainer.set_instance(instance)
            sufficient_reasons = explainer.sufficient_reason(n=3)
            for sr in sufficient_reasons:
                self.assertFalse(explainer.reason_contains_features(sr, 'Origin_Hispanic'))
                self.assertFalse(explainer.reason_contains_features(sr, 'Origin_African_American'))

            contrastives_reasons = explainer.contrastive_reason(n=3)
            for sr in contrastives_reasons:
                self.assertFalse(explainer.reason_contains_features(sr, 'Origin_Hispanic'))
                self.assertFalse(explainer.reason_contains_features(sr, 'Origin_African_American'))


        explainer.set_excluded_features(['Female'])
        for instance, prediction in instances:
            explainer.set_instance(instance)
            sufficient_reasons = explainer.sufficient_reason(n=3)
            for sr in sufficient_reasons:
                self.assertFalse(explainer.reason_contains_features(sr, 'Female'))
        
            contrastives_reasons = explainer.contrastive_reason(n=3)
            for sr in contrastives_reasons:
                self.assertFalse(explainer.reason_contains_features(sr, 'Female'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
