from pyxai import Learning, Explainer, Tools
import math

Tools.set_verbose(0)

import unittest

class TestXGBoost(unittest.TestCase):
    PRECISION = 1

    def test_binary_class(self):
        learner = Learning.Xgboost("tests/dermatology.csv")
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)

        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)
                

    def test_multi_class(self):
        learner = Learning.Xgboost("tests/iris.csv")
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)

        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)

    def test_regression_creditcard(self):
        learner = Learning.Xgboost("tests/creditcard.csv")

        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2, learner_type=Learning.REGRESSION, base_score=0, n_estimators=5)

        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for i, (instance, prediction_classifier) in enumerate(instances):
                
                prediction_classifier = round(float(prediction_classifier),TestXGBoost.PRECISION)
                prediction_model_1 = round(model.predict_instance(instance),TestXGBoost.PRECISION)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = round(model.predict_implicant(implicant),TestXGBoost.PRECISION)
                
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)

    def test_regression_iris(self):
        learner = Learning.Xgboost("tests/iris.csv")

        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2, learner_type=Learning.REGRESSION, base_score=0, n_estimators=5)

        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for i, (instance, prediction_classifier) in enumerate(instances):
                
                prediction_classifier = round(float(prediction_classifier),TestXGBoost.PRECISION)
                prediction_model_1 = round(model.predict_instance(instance),TestXGBoost.PRECISION)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = round(model.predict_implicant(implicant),TestXGBoost.PRECISION)
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)

if __name__ == '__main__':
    unittest.main()