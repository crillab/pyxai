from pyxai import Learning, Explainer, Tools
import math
from decimal import Decimal
from collections.abc import Iterable
import shutil
Tools.set_verbose(0)

import unittest


class TestSaveLoadScikitlearn(unittest.TestCase):
    PRECISION = 3


    def test_hold_out_DT(self):
        self.launch_save("tests/dermatology.csv", Learning.HOLD_OUT, Learning.DT)
        self.launch_load("tests/dermatology.csv")
        self.launch_save("tests/iris.csv", Learning.HOLD_OUT, Learning.DT)
        self.launch_load("tests/iris.csv")
        shutil.rmtree("try_save")

    def test_k_folds_DT(self):
        self.launch_save("tests/dermatology.csv", Learning.K_FOLDS, Learning.DT)
        self.launch_load("tests/dermatology.csv")
        self.launch_save("tests/iris.csv", Learning.K_FOLDS, Learning.DT)
        self.launch_load("tests/iris.csv")
        shutil.rmtree("try_save")

    def test_leave_one_group_out_DT(self):
        self.launch_save("tests/dermatology.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.DT)
        self.launch_load("tests/dermatology.csv")
        self.launch_save("tests/iris.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.DT)
        self.launch_load("tests/iris.csv")
        shutil.rmtree("try_save")

    def test_hold_out_RF(self):
        self.launch_save("tests/dermatology.csv", Learning.HOLD_OUT, Learning.RF)
        self.launch_load("tests/dermatology.csv")
        self.launch_save("tests/iris.csv", Learning.HOLD_OUT, Learning.RF)
        self.launch_load("tests/iris.csv")
        shutil.rmtree("try_save")

    def test_k_folds_RF(self):
        self.launch_save("tests/dermatology.csv", Learning.K_FOLDS, Learning.RF)
        self.launch_load("tests/dermatology.csv")
        self.launch_save("tests/iris.csv", Learning.K_FOLDS, Learning.RF)
        self.launch_load("tests/iris.csv")
        shutil.rmtree("try_save")

    def test_leave_one_group_out_RF(self):
        self.launch_save("tests/dermatology.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.RF)
        self.launch_load("tests/dermatology.csv")
        self.launch_save("tests/iris.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.RF)
        self.launch_load("tests/iris.csv")
        shutil.rmtree("try_save")

    def launch_save(self, dataset, method, output):
        learner = Learning.Scikitlearn(dataset, learner_type=Learning.CLASSIFICATION)
        models = learner.evaluate(method=method, output=output, test_size=0.2)
        learner.save(models, "try_save")


    def launch_load(self, dataset):
        learner, models = Learning.load(models_directory="try_save", tests=True, dataset=dataset)
        if not isinstance(models, Iterable):
            models = [models]
        for model in models:
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)


if __name__ == '__main__':
    print("Tests: " + TestSaveLoadScikitlearn.__name__ + ":")
    unittest.main()
