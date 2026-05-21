from pyxai import Builder, Learning, Explaining, Tools
import unittest

Tools.set_verbose(0)


class TestExplainingRF(unittest.TestCase):
    model = None

    
    def init(cls):
        if cls.model is None:
            cls.learner = Learning.Scikitlearn("tests/datasets/12_fault.csv", problem_type='classification')
            cls.model = cls.learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF, model_parameters={"n_estimators": 10, "max_depth": 4})
        return cls.learner, cls.model


    #@unittest.skip("TO DO")
    def test_sufficient(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=3)
        for instance, _ in instances:
            explainer.set_instance(instance)
            try:
                sufficient_reason = explainer.sufficient_reason(time_limit=5)
            except OSError:
                print('OSError in TestExplainingRF')
                pass
            if explainer.elapsed_time == Explaining.TIMEOUT:
                print("Timeout => no tests to determine if reason is sufficient")
                self.assertTrue(explainer.is_reason(sufficient_reason))
            else:
                self.assertTrue(explainer.is_sufficient_reason(sufficient_reason, n_samples=100) != False)

    #@unittest.skip("TO DO")
    def test_majoritary(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model, features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=10)
        for instance, _ in instances:
            explainer.set_instance(instance)
            majoritary_reason = explainer.majoritary_reason()
            self.assertTrue(explainer.is_majoritary_reason(majoritary_reason, n_samples=100))

    #@unittest.skip("TO DO")
    def test_minimal_majoritary(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)  # ), features_type={"numerical": Learning.DEFAULT})
        instances = learner.get_instances(model, n=10)
        for instance, _ in instances:
            explainer.set_instance(instance)
            majoritary_reason = explainer.minimal_majoritary_reason(time_limit=5)
            if len(majoritary_reason) == 0:
                print("Timeout => No majoritary reason found")
            else:
                self.assertTrue(explainer.is_majoritary_reason(majoritary_reason, n_samples=100))

    #@unittest.skip("TO DO")
    def test_contrastive(self):
        return
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        instances = learner.get_instances(model, n=10)
        for instance, _ in instances:
            explainer.set_instance(instance)
            contrastive_reason = explainer.minimal_contrastive_reason(time_limit=5)
            self.assertTrue(len(contrastive_reason) == 0 or explainer.is_contrastive_reason(contrastive_reason))

    #@unittest.skip("TO DO")
    def test_excluded(self):
        learner, model = self.init()
        explainer = Explaining.initialize(model)
        explainer.set_excluded_features(['African_American', 'Hispanic'])
        instances = learner.get_instances(model, n=2)
        for instance, _ in instances:

            explainer.set_instance(instance)
            try:
                sufficient_reason = explainer.sufficient_reason(time_limit=1)
                self.assertFalse(explainer.reason_contains_features(sufficient_reason, 'Hispanic'))
                self.assertFalse(explainer.reason_contains_features(sufficient_reason, 'African_American'))
            except OSError:
                print('OSError in TestExplainingRF')
                pass
            contrastive_reason = explainer.minimal_contrastive_reason(time_limit=1)
            self.assertTrue(len(contrastive_reason) == 0 or explainer.is_contrastive_reason(contrastive_reason))
            self.assertFalse(explainer.reason_contains_features(contrastive_reason, 'Hispanic'))
            self.assertFalse(explainer.reason_contains_features(contrastive_reason, 'African_American'))

        explainer.set_excluded_features(['Female'])
        for instance, _ in instances:
            explainer.set_instance(instance)
            majoritary_reason = explainer.majoritary_reason()
            self.assertFalse(explainer.reason_contains_features(majoritary_reason, 'Female'))
            contrastive_reason = explainer.minimal_contrastive_reason(time_limit=2)
            self.assertFalse(explainer.reason_contains_features(contrastive_reason, 'Female'))


if __name__ == '__main__':
    unittest.main(verbosity=4)
