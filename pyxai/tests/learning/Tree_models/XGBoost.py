from pyxai import Learning, Tools
import math

Tools.set_verbose(0)

import unittest
import numpy
import xgboost
import json

class TestLearningXGBoost(unittest.TestCase):
    PRECISION = 5

    #@unittest.skip("reason for skipping")
    def test_parameters(self):
        learner = Learning.Xgboost("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION)
        model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT, splitting_parameters={'test_size':0.2}, model_parameters={'max_depth':6, 'base_score':0.5})
        self.assertEqual(model.raw_model.get_xgb_params()["max_depth"], 6)
        self.assertEqual(model.raw_model.get_xgb_params()["base_score"], 0.5)

        models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.BT, splitting_parameters={}, model_parameters={'max_depth':6, 'base_score':0.5})
        for model in models:
            self.assertEqual(model.raw_model.get_xgb_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_xgb_params()["base_score"], 0.5)

        models = learner.evaluate(splitting_method=Learning.LEAVE_ONE_GROUP_OUT, model_type=Learning.BT, splitting_parameters={}, model_parameters={'max_depth':6, 'base_score':0.5})
        for model in models:
            self.assertEqual(model.raw_model.get_xgb_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_xgb_params()["base_score"], 0.5)

        learner = Learning.Xgboost("tests/datasets/dermatology.csv", problem_type=Learning.REGRESSION)
        models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.BT, splitting_parameters={}, model_parameters={'base_score':0.5, 'n_estimators':5})
        for model in models:
            self.assertEqual(len(model.raw_model.get_booster().get_dump()), 5)  # for n_estimators
            self.assertEqual(model.raw_model.get_xgb_params()["base_score"], 0.5)

    #@unittest.skip("Some errors with predictions")
    def test_binary_class(self):
        for base_score in [None, 0.1, 0.5, 0.8]:
            for rnd in range(2):
                learner = Learning.Xgboost("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION)
                models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.BT, 
                                        splitting_parameters={'random_state':rnd}, 
                                        model_parameters={'base_score': base_score, 'random_state':rnd+10})


                for i, model in enumerate(models):
                    instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST, details=True)
                    for detail in instances:
                        instance = detail["instance"]

                        prediction_classifier = detail["prediction"]
                        prediction_model_1 = model.predict_instance(instance)
                        implicant = model.instance_to_binaries(instance)
                        prediction_model_2 = model.predict_implicant(implicant)

                        """ TODO remove when bugs are corrected """
                        raw_model = learner.get_raw_models()[i]
                        #raw_prediction_classifier = raw_model.get_booster().predict(xgboost.DMatrix(numpy.reshape(instance, (1,len(instance))), feature_names=learner.feature_names), pred_leaf=True) 
                        raw_prediction_classifier = raw_model.predict_proba(numpy.reshape(instance, (1,len(instance))))
                        """ """                    
                        
                        self.assertEqual(prediction_classifier, prediction_model_1)
                        self.assertEqual(prediction_classifier, prediction_model_2)
                
    #@unittest.skip("reason for skipping")
    def test_multi_class(self):
        for rnd in range(2):
            learner = Learning.Xgboost("tests/datasets/iris.csv", problem_type=Learning.CLASSIFICATION)
            models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.BT, 
                                      splitting_parameters={'random_state':rnd}, 
                                      model_parameters={'base_score': 0.2, 'random_state':rnd, "n_estimators": 2})

            for model in models:
                instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
                for (instance, prediction_classifier) in instances:
                    prediction_model_1 = model.predict_instance(instance)
                    implicant = model.instance_to_binaries(instance)
                    prediction_model_2 = model.predict_implicant(implicant)
                    self.assertEqual(prediction_classifier, prediction_model_1)
                    self.assertEqual(prediction_classifier, prediction_model_2)
           

    #@unittest.skip("reason for skipping")
    def test_regression_dermatology(self):
        self.regression(Learning.Xgboost("tests/datasets/dermatology.csv", problem_type=Learning.REGRESSION))
    
    #@unittest.skip("reason for skipping")
    def test_regression_wine(self):
        self.regression(Learning.Xgboost("tests/datasets/winequality-red.csv", problem_type=Learning.REGRESSION))

    
    def regression(self, learner):
        models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.BT, 
                                  splitting_parameters={'random_state':0}, 
                                  model_parameters={'random_state':0})#, 'base_score':0.5})

        for j, model in enumerate(models):
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST, details=True)
            raw_model = learner.get_raw_models()[j]
            for i, data in enumerate(instances):
                instance = data["instance"]
                prediction_classifier = data["prediction"]
                
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                
                self.assertAlmostEqual(prediction_classifier, prediction_model_1, places=TestLearningXGBoost.PRECISION)
                self.assertAlmostEqual(prediction_classifier, prediction_model_2, places=TestLearningXGBoost.PRECISION)

if __name__ == '__main__':
    print("Tests: " + TestLearningXGBoost.__name__ + ":")
    unittest.main()
