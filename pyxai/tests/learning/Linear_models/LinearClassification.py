from pyxai import Learning, Tools
import math
import pandas

Tools.set_verbose(0)

import unittest


class TestLearningLinearClassification(unittest.TestCase):
    PRECISION = 4

    def test_ridge_classification_dermatology(self):
        self.ridge_classification(Learning.Scikitlearn("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION))

    def test_logistic_classification_dermatology(self):
        self.logistic_classification(Learning.Scikitlearn("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION))


    def test_ridge_classification_iris(self):
        self.ridge_classification(Learning.Scikitlearn("tests/datasets/iris.csv", problem_type=Learning.CLASSIFICATION))

    def test_logistic_classification_iris(self):
        self.logistic_classification(Learning.Scikitlearn("tests/datasets/iris.csv", problem_type=Learning.CLASSIFICATION))
        
    def ridge_classification(self, learner):
        models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.LINEAR_RIDGE, splitting_parameters={}, model_parameters={'alpha':0.5})

        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for i, (instance, prediction) in enumerate(instances):
                prediction_raw_model = model.learner_information.raw_model.predict(pandas.DataFrame([instance]))
                self.assertEqual(prediction, prediction_raw_model)   

    def logistic_classification(self, learner):
        models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.LINEAR_LOGISTIC, splitting_parameters={}, model_parameters={'max_iter': 1000})

        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for i, (instance, prediction) in enumerate(instances):
                prediction_raw_model = model.learner_information.raw_model.predict(pandas.DataFrame([instance]))
                self.assertEqual(prediction, prediction_raw_model)  

if __name__ == '__main__':
    print("Tests: " + TestLearningLinearClassification.__name__ + ":")
    unittest.main()
