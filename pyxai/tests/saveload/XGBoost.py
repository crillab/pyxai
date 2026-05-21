from pyxai import Learning, Tools
import math
from decimal import Decimal

Tools.set_verbose(0)
from collections.abc import Iterable
import unittest
import shutil

class TestSaveLoadXgboost(unittest.TestCase):
    PRECISION = 3

    #@unittest.skip("reason for skipping")
    def test_hold_out_classification(self):
        self.launch_save("tests/datasets/dermatology.csv", Learning.HOLD_OUT, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/datasets/dermatology.csv")
        self.launch_save("tests/datasets/iris.csv", Learning.HOLD_OUT, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/datasets/iris.csv")
        shutil.rmtree("try_save")

    #@unittest.skip("reason for skipping")
    def test_k_folds_classification(self):
        self.launch_save("tests/datasets/dermatology.csv", Learning.K_FOLDS, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/datasets/dermatology.csv")
        self.launch_save("tests/datasets/iris.csv", Learning.K_FOLDS, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/datasets/iris.csv")
        shutil.rmtree("try_save")

    #@unittest.skip("reason for skipping")
    def test_leave_one_group_out_classification(self):
        self.launch_save("tests/datasets/dermatology.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/datasets/dermatology.csv")
        self.launch_save("tests/datasets/iris.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/datasets/iris.csv")
        shutil.rmtree("try_save")

    #@unittest.skip("reason for skipping")
    def test_hold_out_regression(self):
        self.launch_save("tests/datasets/winequality-red.csv", Learning.HOLD_OUT, Learning.BT, Learning.REGRESSION)
        self.launch_load("tests/datasets/winequality-red.csv")
        shutil.rmtree("try_save")

    #@unittest.skip("reason for skipping")
    def test_k_folds_regression(self):
        self.launch_save("tests/datasets/winequality-red.csv", Learning.K_FOLDS, Learning.BT, Learning.REGRESSION)
        self.launch_load("tests/datasets/winequality-red.csv")
        shutil.rmtree("try_save")

    #@unittest.skip("reason for skipping")
    def test_leave_one_group_out_regression(self):
        self.launch_save("tests/datasets/winequality-red.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.BT, Learning.REGRESSION)
        self.launch_load("tests/datasets/winequality-red.csv", precision=1)
        shutil.rmtree("try_save")

    #@unittest.skip("reason for skipping")
    def test_continuity_launcher(self):
        self.save_load_continuity('tests/datasets/winequality-red.csv', 'regression', 'boosted-tree', 'discrete-values')
        self.save_load_continuity('tests/datasets/winequality-red.csv', 'regression', 'boosted-tree', 'continuous-values')
        self.save_load_continuity('tests/datasets/iris.csv', 'classification', 'boosted-tree', None)
        self.save_load_continuity('tests/datasets/weatherAUS_simplified.csv', 'classification', 'boosted-tree', None)
        shutil.rmtree("try_save")

    def launch_save(self, dataset, method, output, problem_type):
        learner = Learning.Xgboost(dataset, problem_type=problem_type)
        models = learner.evaluate(splitting_method=method, model_type=output, splitting_parameters={})
        Learning.ModelIO.save(models, "try_save")


    def launch_load(self, dataset, precision=None):
        if precision is None:
            precision = TestSaveLoadXgboost.PRECISION
        learner, models = Learning.ModelIO.load(models_directory="try_save", tests=True, data=dataset)
        if not isinstance(models, Iterable):
            models = [models]
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TRAIN)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                if learner.problem_type == Learning.REGRESSION:
                    str_f = "{:." + str(precision) + "f}"
                    prediction_classifier = str_f.format(prediction_classifier)
                    prediction_model_1 = str_f.format(prediction_model_1)
                    prediction_model_2 = str_f.format(prediction_model_2)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)


    def save_load_continuity(self, dataset, problem_type, model_type, labels_type):
        n_instances = 10
        learner = Learning.Xgboost(dataset, problem_type=problem_type, labels_type=labels_type)
        model = learner.evaluate(splitting_method='hold-out', model_type=model_type, splitting_parameters={"test_size":0.2, "random_state":0})
        instances_initial = learner.get_instances(model=model, n=n_instances, indexes=Learning.TEST)

        Learning.ModelIO.save(model, "try_save")
        del learner
        del model
        learner, model = Learning.ModelIO.load(models_directory="try_save", tests=True, data=dataset)
        instances_load = learner.get_instances(model=model, n=n_instances, indexes=Learning.TEST)

        for i in range(n_instances):
            self.assertEqual(instances_initial[i][1], instances_load[i][1])


if __name__ == '__main__':
    print("Tests: " + TestSaveLoadXgboost.__name__ + ":")
    unittest.main()
