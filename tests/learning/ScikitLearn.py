from pyxai import Learning, Explainer, Tools
import math

Tools.set_verbose(0)

import unittest

class TestXGBoost(unittest.TestCase):
    PRECISION = 1

    def test_parameters(self):
        learner = Learning.Scikitlearn("tests/dermatology.csv")
        model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, test_size=0.2, max_depth=6)
        self.assertEqual(model.raw_model.get_params()["max_depth"],6)
        self.assertEqual(model.raw_model.get_params()["random_state"],0)

        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.RF, test_size=0.2, max_depth=6)
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"],6)
            self.assertEqual(model.raw_model.get_params()["random_state"],0)

        models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.RF, test_size=0.2, max_depth=6)
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"],6)
            self.assertEqual(model.raw_model.get_params()["random_state"],0)
    
        learner = Learning.Scikitlearn("tests/iris.csv")
        model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT, test_size=0.2, max_depth=6)
        self.assertEqual(model.raw_model.get_params()["max_depth"],6)
        self.assertEqual(model.raw_model.get_params()["random_state"],0)

        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.DT, test_size=0.2, max_depth=6)
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"],6)
            self.assertEqual(model.raw_model.get_params()["random_state"],0)

        models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.DT, test_size=0.2, max_depth=6)
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"],6)
            self.assertEqual(model.raw_model.get_params()["random_state"],0)

    def test_prediction(self):
        self.prediction(Learning.Scikitlearn("tests/dermatology.csv"))
        self.prediction(Learning.Scikitlearn("tests/iris.csv"))
        
        #self.bug(Learning.Scikitlearn("tests/iris.csv"))
    
    def bug(self, learner):
        models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.DT, test_size=0.2)
        
        for model in [models[8]]:
            from sklearn.tree import export_text
            print(export_text(model.raw_model))
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            instance = instances[4][0]
            prediction_classifier = instances[4][1]
            instance = [6.,  3,  4.8, 1.8]
            prediction_model_1 = model.predict_instance(instance)
            implicant = model.instance_to_binaries(instance)
            prediction_model_2 = model.predict_implicant(implicant)
            print("instance:", instance)
            print("prediction_classifier:", prediction_classifier)
            print("prediction_model_1:", prediction_model_1)
            print("prediction_model_2:", prediction_model_2)
            self.assertEqual(prediction_classifier,prediction_model_1)
            self.assertEqual(prediction_classifier,prediction_model_2)

    def prediction(self, learner):
        model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT, test_size=0.2)
        instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
        for (instance, prediction_classifier) in instances:
            prediction_model_1 = model.predict_instance(instance)
            implicant = model.instance_to_binaries(instance)
            prediction_model_2 = model.predict_implicant(implicant)
            self.assertEqual(prediction_classifier,prediction_model_1)
            self.assertEqual(prediction_classifier,prediction_model_2)

        models = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, test_size=0.2)
        instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
        for (instance, prediction_classifier) in instances:
            prediction_model_1 = model.predict_instance(instance)
            implicant = model.instance_to_binaries(instance)
            prediction_model_2 = model.predict_implicant(implicant)
            self.assertEqual(prediction_classifier,prediction_model_1)
            self.assertEqual(prediction_classifier,prediction_model_2)
        
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.DT, test_size=0.2)
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)

        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.RF, test_size=0.2)
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)
        
        models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.DT, test_size=0.2)
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)

        models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.RF, test_size=0.2)
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)
                

if __name__ == '__main__':
    unittest.main()