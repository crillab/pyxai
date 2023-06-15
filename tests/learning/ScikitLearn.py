from pyxai import Learning, Explainer, Tools
import math

Tools.set_verbose(0)

import unittest


class TestLearningScikitlearn(unittest.TestCase):
    PRECISION = 1


    def test_parameters(self):
        learner = Learning.Scikitlearn("tests/dermatology.csv", learner_type=Learning.CLASSIFICATION)
        model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, test_size=0.2, max_depth=6)
        self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
        self.assertEqual(model.raw_model.get_params()["random_state"], 0)

        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.RF, test_size=0.2, max_depth=6)
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_params()["random_state"], 0)

        models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.RF, test_size=0.2, max_depth=6)
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_params()["random_state"], 0)

        learner = Learning.Scikitlearn("tests/iris.csv", learner_type=Learning.CLASSIFICATION)
        model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT, test_size=0.2, max_depth=6)
        self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
        self.assertEqual(model.raw_model.get_params()["random_state"], 0)

        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.DT, test_size=0.2, max_depth=6)
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_params()["random_state"], 0)

        models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.DT, test_size=0.2, max_depth=6)
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_params()["random_state"], 0)


    def test_prediction_dermatology(self):
        self.prediction(Learning.Scikitlearn("tests/dermatology.csv", learner_type=Learning.CLASSIFICATION))


    def test_prediction_iris(self):
        self.prediction(Learning.Scikitlearn("tests/iris.csv", learner_type=Learning.CLASSIFICATION))


    def prediction(self, learner):
        model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT, test_size=0.2)
        instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
        for (instance, prediction_classifier) in instances:
            prediction_model_1 = model.predict_instance(instance)
            implicant = model.instance_to_binaries(instance)
            prediction_model_2 = model.predict_implicant(implicant)
            self.assertEqual(prediction_classifier, prediction_model_1)
            self.assertEqual(prediction_classifier, prediction_model_2)

        models = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, test_size=0.2)
        instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
        for (instance, prediction_classifier) in instances:
            prediction_model_1 = model.predict_instance(instance)
            implicant = model.instance_to_binaries(instance)
            prediction_model_2 = model.predict_implicant(implicant)
            self.assertEqual(prediction_classifier, prediction_model_1)
            self.assertEqual(prediction_classifier, prediction_model_2)

        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.DT, test_size=0.2)
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)

        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.RF, test_size=0.2)
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)

        models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.DT, test_size=0.2)
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)

        models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.RF, test_size=0.2)
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)


if __name__ == '__main__':
    print("Tests: " + TestLearningScikitlearn.__name__ + ":")
    unittest.main()
