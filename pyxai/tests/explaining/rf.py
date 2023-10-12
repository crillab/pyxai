from pyxai import Builder, Learning, Explainer, Tools
import unittest

Tools.set_verbose(0)


class TestRF(unittest.TestCase):
    model = None


    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Scikitlearn("tests/compas.csv", learner_type=Learning.CLASSIFICATION)
            cls.model = cls.learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
        return cls.learner, cls.model


    def test_sufficient(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=30)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            sufficient_reason = explainer.sufficient_reason(time_limit=5)
            #if explainer.elapsed_time == Explainer.TIMEOUT:
            #    self.assertTrue(explainer.is_reason(sufficient_reason))
            #else:
            #    self.assertTrue(explainer.is_sufficient_reason(sufficient_reason) != False)


    def test_majoritary(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=30)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            majoritary_reason = explainer.majoritary_reason()
            self.assertTrue(explainer.is_majoritary_reason(majoritary_reason))


    def test_minimal_majoritary(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)  # ), features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=10)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            majoritary_reason = explainer.minimal_majoritary_reason(time_limit=5)
            self.assertTrue(len(majoritary_reason) == 0 or explainer.is_majoritary_reason(majoritary_reason))


    def test_contrastive(self):
        return
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        instances = learner.get_instances(model, n=10)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            contrastive_reason = explainer.minimal_contrastive_reason(time_limit=5)
            self.assertTrue(len(contrastive_reason) == 0 or explainer.is_contrastive_reason(contrastive_reason))


    def test_excluded(self):
        learner, model = self.init()
        explainer = Explainer.initialize(model)
        explainer.set_excluded_features(['African_American', 'Hispanic'])
        instances = learner.get_instances(model, n=10)
        for instance, prediction in instances:
            explainer.set_instance(instance)
            sufficient_reason = explainer.sufficient_reason()
            self.assertFalse(explainer.reason_contains_features(sufficient_reason, 'Hispanic'))
            self.assertFalse(explainer.reason_contains_features(sufficient_reason, 'African_American'))

            contrastives_reasons = explainer.minimal_contrastive_reason(time_limit=2)
            for sr in contrastives_reasons:
                self.assertFalse(explainer.reason_contains_features(sr, 'Hispanic'))
                self.assertFalse(explainer.reason_contains_features(sr, 'African_American'))

        explainer.set_excluded_features(['Female'])
        for instance, prediction in instances:
            explainer.set_instance(instance)
            majoritary_reason = explainer.majoritary_reason()
            self.assertFalse(explainer.reason_contains_features(majoritary_reason, 'Female'))
            contrastive_reason = explainer.minimal_contrastive_reason(time_limit=2)
            self.assertFalse(explainer.reason_contains_features(contrastive_reason, 'Female'))


if __name__ == '__main__':
    unittest.main(verbosity=3)
