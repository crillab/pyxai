from pyxai import Learning, Explainer, Tools
import math

Tools.set_verbose(0)

import unittest

class TestXGBoost(unittest.TestCase):
    PRECISION = 1
    def setUp(self):
        print("..|In method:", self._testMethodName)

    def test_1_save_classification(self):
        learner = Learning.Xgboost("tests/iris.csv", learner_type=Learning.CLASSIFICATION)
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)
        learner.save(models, "try_save")

    def test_2_load_classification(self):    
        learner, models = Learning.load(models_directory="try_save", tests=True, dataset="tests/iris.csv") 

    def test_3_save_regression(self):
        learner = Learning.Xgboost("tests/winequality-red.csv", learner_type=Learning.REGRESSION)
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)
        learner.save(models, "try_save")

    def test_4_load_regression(self):    
        learner, models = Learning.load(models_directory="try_save", tests=True, dataset="tests/winequality-red.csv") 
   
if __name__ == '__main__':
    unittest.main()