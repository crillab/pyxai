from pyxai import Learning, Explainer, Tools
import math
from decimal import Decimal
Tools.set_verbose(0)
from collections.abc import Iterable
import unittest

class TestSaveLoadLightGBM(unittest.TestCase):
    
    def setUp(self):
        print("..|In method:", self._testMethodName)

    

    def test_HOLD_OUT_REGRESSION(self):
        self.launch_save("tests/winequality-red.csv", Learning.HOLD_OUT, Learning.BT, Learning.REGRESSION)
        self.launch_load("tests/winequality-red.csv", Learning.REGRESSION)
    
    def test_K_FOLDS_REGRESSION(self):
        self.launch_save("tests/winequality-red.csv", Learning.K_FOLDS, Learning.BT, Learning.REGRESSION)
        self.launch_load("tests/winequality-red.csv", Learning.REGRESSION)
    
    def test_LEAVE_ONE_GROUP_OUT_REGRESSION(self):
        self.launch_save("tests/winequality-red.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.BT, Learning.REGRESSION)
        self.launch_load("tests/winequality-red.csv", Learning.REGRESSION)

    def launch_save(self, dataset, method, output, learner_type):
        learner = Learning.LightGBM(dataset, learner_type=learner_type)
        models = learner.evaluate(method=method, output=output, test_size=0.2)
        learner.save(models, "try_save")

    def launch_load(self, dataset, learner_type):
        learner, models = Learning.load(models_directory="try_save", tests=True, dataset=dataset) 
        if not isinstance(models, Iterable):
            models = [models]
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)

if __name__ == '__main__':
    print("Tests: " + TestSaveLoadLightGBM.__name__ + ":")
    unittest.main()