from pyxai import Learning, Explainer, Tools
import math

Tools.set_verbose(0)

import unittest


class TestLearningLightGBM(unittest.TestCase):

    def test_parameters(self):
        learner = Learning.LightGBM("tests/dermatology.csv", learner_type=Learning.REGRESSION)
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2, learning_rate=0.3, n_estimators=5)
        for model in models:
            self.assertEqual(model.raw_model.booster_.num_trees(), 5)  # for n_estimators
            self.assertEqual(model.raw_model.get_params()["learning_rate"], 0.3)



    def test_regression_dermatology(self):
        self.regression(Learning.LightGBM("tests/dermatology.csv", learner_type=Learning.REGRESSION))


    def test_regression_wine(self):
        self.regression(Learning.LightGBM("tests/winequality-red.csv", learner_type=Learning.REGRESSION))


    def regression(self, learner):
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for i, (instance, prediction_classifier) in enumerate(instances):
                prediction_classifier = float(prediction_classifier)
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)


if __name__ == '__main__':
    print("Tests: " + TestLearningLightGBM.__name__ + ":")
    unittest.main()
