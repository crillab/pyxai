from pyxai import Learning, Tools
import math
from decimal import Decimal
from collections.abc import Iterable
import shutil
Tools.set_verbose(0)

import unittest


class TestSaveLoadScikitlearn(unittest.TestCase):
    PRECISION = 3


    def test_hold_out_DT(self):
        self.launch_save("tests/datasets/dermatology.csv", Learning.HOLD_OUT, Learning.DT)
        self.launch_load("tests/datasets/dermatology.csv")
        self.launch_save("tests/datasets/iris.csv", Learning.HOLD_OUT, Learning.DT)
        self.launch_load("tests/datasets/iris.csv")
        shutil.rmtree("try_save")

    def test_k_folds_DT(self):
        self.launch_save("tests/datasets/dermatology.csv", Learning.K_FOLDS, Learning.DT)
        self.launch_load("tests/datasets/dermatology.csv")
        self.launch_save("tests/datasets/iris.csv", Learning.K_FOLDS, Learning.DT)
        self.launch_load("tests/datasets/iris.csv")
        shutil.rmtree("try_save")

    def test_leave_one_group_out_DT(self):
        self.launch_save("tests/datasets/dermatology.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.DT)
        self.launch_load("tests/datasets/dermatology.csv")
        self.launch_save("tests/datasets/iris.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.DT)
        self.launch_load("tests/datasets/iris.csv")
        shutil.rmtree("try_save")

    def test_hold_out_RF(self):
        self.launch_save("tests/datasets/dermatology.csv", Learning.HOLD_OUT, Learning.RF)
        self.launch_load("tests/datasets/dermatology.csv")
        self.launch_save("tests/datasets/iris.csv", Learning.HOLD_OUT, Learning.RF)
        self.launch_load("tests/datasets/iris.csv")
        shutil.rmtree("try_save")

    def test_k_folds_RF(self):
        self.launch_save("tests/datasets/dermatology.csv", Learning.K_FOLDS, Learning.RF)
        self.launch_load("tests/datasets/dermatology.csv")
        self.launch_save("tests/datasets/iris.csv", Learning.K_FOLDS, Learning.RF)
        self.launch_load("tests/datasets/iris.csv")
        shutil.rmtree("try_save")

    def test_leave_one_group_out_RF(self):
        self.launch_save("tests/datasets/dermatology.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.RF)
        self.launch_load("tests/datasets/dermatology.csv")
        self.launch_save("tests/datasets/iris.csv", Learning.LEAVE_ONE_GROUP_OUT, Learning.RF)
        self.launch_load("tests/datasets/iris.csv")
        shutil.rmtree("try_save")

    def test_continuity_launcher(self):
        self.save_load_continuity('tests/datasets/winequality-red.csv', 'regression', 'linear-simple', 'discrete-values')
        self.save_load_continuity('tests/datasets/winequality-red.csv', 'regression', 'linear-ridge', 'continuous-values')
        self.save_load_continuity('tests/datasets/iris.csv', 'classification', 'decision-tree', None)
        self.save_load_continuity('tests/datasets/weatherAUS_simplified.csv', 'classification', 'decision-tree', None)
        shutil.rmtree("try_save")

    def save_load_continuity(self, dataset, problem_type, model_type, labels_type):
        n_instances = 10
        learner = Learning.Scikitlearn(dataset, problem_type=problem_type, labels_type=labels_type)
        model = learner.evaluate(splitting_method='hold-out', model_type=model_type, splitting_parameters={"test_size":0.2, "random_state":0})
        instances_initial = learner.get_instances(model=model, n=n_instances, indexes=Learning.TEST)

        Learning.ModelIO.save(model, "try_save")
        
        #learner.save(model, "try_save")
        del learner
        del model
        learner, model = Learning.ModelIO.load(models_directory="try_save", tests=True, data=dataset)
        instances_load = learner.get_instances(model=model, n=n_instances, indexes=Learning.TEST)

        for i in range(n_instances):
            self.assertEqual(instances_initial[i][1], instances_load[i][1])

    def launch_save(self, dataset, method, output):
        learner = Learning.Scikitlearn(dataset, problem_type=Learning.CLASSIFICATION)
        models = learner.evaluate(splitting_method=method, model_type=output, splitting_parameters={})
        Learning.ModelIO.save(models, "try_save")


    def launch_load(self, dataset):
        learner, models = Learning.ModelIO.load(models_directory="try_save", tests=True, data=dataset)
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
