from pyxai import Learning, Tools
import math
import pandas

Tools.set_verbose(0)

import unittest


class TestLearningLinearRegression(unittest.TestCase):
    PRECISION = 10

    def test_simple_regression_dermatology(self):
        self.main("tests/datasets/dermatology.csv", "Regression", "linear-simple")

    def test_simple_regression_wine(self):
        self.main("tests/datasets/winequality-red.csv", Learning.REGRESSION, Learning.LINEAR_SIMPLE)

    
    def test_ridge_regression_dermatology(self):
        self.main("tests/datasets/dermatology.csv", "regression", "linear-ridge",
                  model_parameters = {'alpha':0.5})

    def test_ridge_regression_wine(self):
        self.main("tests/datasets/winequality-red.csv", Learning.REGRESSION, Learning.LINEAR_RIDGE,
                  model_parameters = {'alpha':0.5})


    
    def main(self, dataset, problem_type, model_type, splitting_parameters={}, model_parameters={}):
        learner = Learning.Scikitlearn(dataset, problem_type=problem_type)
        models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=model_type,
                                  splitting_parameters=splitting_parameters, model_parameters=model_parameters)

        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for i, (instance, prediction) in enumerate(instances):
                prediction_raw_model = model.learner_information.raw_model.predict(pandas.DataFrame([instance]))[0]
                self.assertAlmostEqual(prediction, prediction_raw_model, TestLearningLinearRegression.PRECISION)


if __name__ == '__main__':
    print("Tests: " + TestLearningLinearRegression.__name__ + ":")
    unittest.main()
