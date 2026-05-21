from pyxai import Learning, Tools
import math
import pandas

Tools.set_verbose(0)

import unittest


class TestLearningScikitlearn(unittest.TestCase):
    #@unittest.skip("reason for skipping")
    def test_parameters(self):
        learner = Learning.Scikitlearn("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION)
        model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF, splitting_parameters={'test_size':0.2, 'random_state':0}, model_parameters={'max_depth':6, 'random_state':0})
        self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
        self.assertEqual(model.raw_model.get_params()["random_state"], 0)

        models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.RF, splitting_parameters={'random_state':0}, model_parameters={'max_depth':6, 'random_state':0})
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_params()["random_state"], 0)

        models = learner.evaluate(splitting_method=Learning.LEAVE_ONE_GROUP_OUT, model_type=Learning.RF, splitting_parameters={'random_state':0}, model_parameters={'max_depth':6, 'random_state':0})
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_params()["random_state"], 0)

        learner = Learning.Scikitlearn("tests/datasets/iris.csv", problem_type=Learning.CLASSIFICATION)
        model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.DT, splitting_parameters={'test_size':0.2, 'random_state':0}, model_parameters={'max_depth':6, 'random_state':0})
        self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
        self.assertEqual(model.raw_model.get_params()["random_state"], 0)

        models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.DT, splitting_parameters={'random_state':0}, model_parameters={'max_depth':6, 'random_state':0})
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_params()["random_state"], 0)

        models = learner.evaluate(splitting_method=Learning.LEAVE_ONE_GROUP_OUT, model_type=Learning.DT, splitting_parameters={'random_state':0}, model_parameters={'max_depth':6, 'random_state':0})
        for model in models:
            self.assertEqual(model.raw_model.get_params()["max_depth"], 6)
            self.assertEqual(model.raw_model.get_params()["random_state"], 0)

    
    #@unittest.skip("reason for skipping")
    def test_prediction_inverse(self):
        df = pandas.read_csv("tests/datasets/dermatology.csv")
        # Get the last column name
        last_col = df.columns[-1]
        # Invert the values (0 becomes 1, 1 becomes 0)
        df[last_col] = 1 - df[last_col]

        self.prediction(Learning.Scikitlearn(df, problem_type=Learning.CLASSIFICATION))

    
    #@unittest.skip("reason for skipping")
    def test_prediction_dermatology(self):
        self.prediction(Learning.Scikitlearn("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION))

    
    #@unittest.skip("reason for skipping")
    def test_prediction_iris(self):
        self.prediction(Learning.Scikitlearn("tests/datasets/iris.csv", problem_type=Learning.CLASSIFICATION))


    def prediction(self, learner):
        model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.DT, splitting_parameters={'test_size':0.2, 'random_state':0}, model_parameters={'random_state':0})
        instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
        for (instance, prediction_classifier) in instances:
            prediction_model_1 = model.predict_instance(instance)
            implicant = model.instance_to_binaries(instance)
            prediction_model_2 = model.predict_implicant(implicant)
            self.assertEqual(prediction_classifier, prediction_model_1)
            self.assertEqual(prediction_classifier, prediction_model_2)

        models = learner.evaluate(splitting_method='hold-out', model_type='random-forest', splitting_parameters={'test_size':0.2, 'random_state':0}, model_parameters={'random_state':0})
        instances = learner.get_instances(model=model, n=10, indexes='test')
        for (instance, prediction_classifier) in instances:
            prediction_model_1 = model.predict_instance(instance)
            implicant = model.instance_to_binaries(instance)
            prediction_model_2 = model.predict_implicant(implicant)
            self.assertEqual(prediction_classifier, prediction_model_1)
            self.assertEqual(prediction_classifier, prediction_model_2)

        models = learner.evaluate(splitting_method='k-folds', model_type='decision-tree', splitting_parameters={'n_models':3, 'random_state':0}, model_parameters={'random_state':0})
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes='test')
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)

        models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.RF, splitting_parameters={'n_models':2, 'random_state':0}, model_parameters={'random_state':0})
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)

        models = learner.evaluate(splitting_method=Learning.LEAVE_ONE_GROUP_OUT, model_type=Learning.DT, splitting_parameters={'n_models':2, 'random_state':0}, model_parameters={'random_state':0})
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)

        models = learner.evaluate(splitting_method=Learning.LEAVE_ONE_GROUP_OUT, model_type=Learning.RF, splitting_parameters={'n_models':6, 'random_state':0}, model_parameters={'random_state':0})
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
