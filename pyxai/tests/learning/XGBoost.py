from pyxai import Learning, Explainer, Tools
import math

Tools.set_verbose(0)

import unittest


class TestLearningXGBoost(unittest.TestCase):
    PRECISION = 4


    def test_parameters(self):
        learner = Learning.Xgboost("tests/dermatology.csv", learner_type=Learning.CLASSIFICATION)
        model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT, test_size=0.2, max_depth=6, base_score=0.5)
        self.assertEqual(model.raw_model.get_xgb_params()["max_depth"], 6)
        self.assertEqual(model.raw_model.get_xgb_params()["base_score"], 0.5)

        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2, max_depth=6, base_score=0.5)
        for model in models:
            self.assertEqual(model.raw_model.get_xgb_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_xgb_params()["base_score"], 0.5)

        models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.BT, test_size=0.2, max_depth=6, base_score=0.5)
        for model in models:
            self.assertEqual(model.raw_model.get_xgb_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_xgb_params()["base_score"], 0.5)

        learner = Learning.Xgboost("tests/dermatology.csv", learner_type=Learning.REGRESSION)
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2, base_score=0, n_estimators=5)
        for model in models:
            self.assertEqual(len(model.raw_model.get_booster().get_dump()), 5)  # for n_estimators
            self.assertEqual(model.raw_model.get_xgb_params()["base_score"], 0)


    def test_binary_class(self):
        learner = Learning.Xgboost("tests/dermatology.csv", learner_type=Learning.CLASSIFICATION)
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)

        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)


    def test_multi_class(self):
        learner = Learning.Xgboost("tests/iris.csv", learner_type=Learning.CLASSIFICATION)
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)

        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)


    

    def test_regression_dermatology(self):
        self.regression(Learning.Xgboost("tests/dermatology.csv", learner_type=Learning.REGRESSION))


    def test_regression_wine(self):
        self.regression(Learning.Xgboost("tests/winequality-red.csv", learner_type=Learning.REGRESSION))


    def regression(self, learner):
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2, base_score=0, n_estimators=5)

        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for i, (instance, prediction_classifier) in enumerate(instances):
                prediction_classifier = round(float(prediction_classifier), TestLearningXGBoost.PRECISION)
                prediction_model_1 = round(model.predict_instance(instance), TestLearningXGBoost.PRECISION)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = round(model.predict_implicant(implicant), TestLearningXGBoost.PRECISION)

                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)


if __name__ == '__main__':
    print("Tests: " + TestLearningXGBoost.__name__ + ":")
    unittest.main()
