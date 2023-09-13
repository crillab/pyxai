from pyxai import Learning, Explainer, Tools
import math
from decimal import Decimal

Tools.set_verbose(0)
from collections.abc import Iterable
import unittest
import shutil

class TestSaveLoadXgboost(unittest.TestCase):
    PRECISION = 3


    def test_hold_out_classification(self):
        self.launch_save("tests/dermatology.csv", Learning.HOLD_OUT, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/dermatology.csv", Learning.CLASSIFICATION)
        self.launch_save("tests/iris.csv", Learning.HOLD_OUT, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/iris.csv", Learning.CLASSIFICATION)
        shutil.rmtree("try_save")

    def test_k_folds_classification(self):
        self.launch_save("tests/dermatology.csv", Learning.K_FOLDS, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/dermatology.csv", Learning.CLASSIFICATION)
        self.launch_save("tests/iris.csv", Learning.K_FOLDS, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/iris.csv", Learning.CLASSIFICATION)
        shutil.rmtree("try_save")

    def test_leave_one_group_out_classification(self):
        self.launch_save("tests/dermatology.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/dermatology.csv", Learning.CLASSIFICATION)
        self.launch_save("tests/iris.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.BT, Learning.CLASSIFICATION)
        self.launch_load("tests/iris.csv", Learning.CLASSIFICATION)
        shutil.rmtree("try_save")

    def test_hold_out_regression(self):
        self.launch_save("tests/winequality-red.csv", Learning.HOLD_OUT, Learning.BT, Learning.REGRESSION)
        self.launch_load("tests/winequality-red.csv", Learning.REGRESSION)
        shutil.rmtree("try_save")

    def test_k_folds_regression(self):
        self.launch_save("tests/winequality-red.csv", Learning.K_FOLDS, Learning.BT, Learning.REGRESSION)
        self.launch_load("tests/winequality-red.csv", Learning.REGRESSION)
        shutil.rmtree("try_save")

    def test_leave_one_group_out_regression(self):
        self.launch_save("tests/winequality-red.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.BT, Learning.REGRESSION)
        self.launch_load("tests/winequality-red.csv", Learning.REGRESSION)
        shutil.rmtree("try_save")

    def launch_save(self, dataset, method, output, learner_type):
        learner = Learning.Xgboost(dataset, learner_type=learner_type)
        models = learner.evaluate(method=method, output=output, test_size=0.2)
        learner.save(models, "try_save")


    def launch_load(self, dataset, learner_type):
        learner, models = Learning.load(models_directory="try_save", tests=True, dataset=dataset)
        if not isinstance(models, Iterable):
            models = [models]
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TRAINING)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                if learner_type == Learning.REGRESSION:
                    str_f = "{:." + str(TestSaveLoadXgboost.PRECISION) + "f}"
                    prediction_classifier = str_f.format(prediction_classifier)
                    prediction_model_1 = str_f.format(prediction_model_1)
                    prediction_model_2 = str_f.format(prediction_model_2)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)


if __name__ == '__main__':
    print("Tests: " + TestSaveLoadXgboost.__name__ + ":")
    unittest.main()
