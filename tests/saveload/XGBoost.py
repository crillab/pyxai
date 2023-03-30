from pyxai import Learning, Explainer, Tools
import math
from decimal import Decimal
Tools.set_verbose(0)

import unittest

class TestXGBoost(unittest.TestCase):
    PRECISION = 3
    def setUp(self):
        print("..|In method:", self._testMethodName)

    def test_1_save_classification(self):
        learner = Learning.Xgboost("tests/iris.csv", learner_type=Learning.CLASSIFICATION)
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)
        learner.save(models, "try_save")

    def test_2_load_classification(self):    
        learner, models = Learning.load(models_directory="try_save", tests=True, dataset="tests/iris.csv") 
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)

    def test_3_save_regression(self):
        learner = Learning.Xgboost("tests/winequality-red.csv", learner_type=Learning.REGRESSION)
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TRAINING)
            for i, (instance, prediction_classifier) in enumerate(instances):
                implicant = model.instance_to_binaries(instance)
                
                str_f = "{:."+str(TestXGBoost.PRECISION)+"f}"
                #print("prediction_classifier:", prediction_classifier)
                #print("model.predict_implicant(implicant):", model.predict_implicant(implicant))
                
                prediction_classifier = str_f.format(prediction_classifier)
                prediction_model_1 = str_f.format(model.predict_instance(instance))
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = str_f.format(model.predict_implicant(implicant))
                
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)
        learner.save(models, "try_save")

    def test_4_load_regression(self):    
        learner, models = Learning.load(models_directory="try_save", tests=True, dataset="tests/winequality-red.csv") 
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TRAINING)
            for i, (instance, prediction_classifier) in enumerate(instances):
                implicant = model.instance_to_binaries(instance)
                
                str_f = "{:."+str(TestXGBoost.PRECISION)+"f}"
                prediction_classifier = str_f.format(prediction_classifier)
                prediction_model_1 = str_f.format(model.predict_instance(instance))
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = str_f.format(model.predict_implicant(implicant))
                
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)
   
if __name__ == '__main__':
    unittest.main()