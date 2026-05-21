from pyxai import Learning, Tools
import math
import shutil
from decimal import Decimal

Tools.set_verbose(0)
from collections.abc import Iterable
import unittest


class TestSaveLoadLightGBM(unittest.TestCase):
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
        self.launch_load("tests/datasets/winequality-red.csv")
        shutil.rmtree("try_save")

    def test_continuity_launcher(self):
        self.save_load_continuity('tests/datasets/winequality-red.csv', 'regression', 'boosted-tree', 'discrete-values')
        self.save_load_continuity('tests/datasets/winequality-red.csv', 'regression', 'boosted-tree', 'continuous-values')
        shutil.rmtree("try_save")

    
    def launch_save(self, dataset, method, output, problem_type):
        learner = Learning.LightGBM(dataset, problem_type=problem_type)
        models = learner.evaluate(splitting_method=method, model_type=output, splitting_parameters={})
        Learning.ModelIO.save(models, "try_save")

    
    def launch_load(self, dataset):
        learner, models = Learning.ModelIO.load(models_directory="try_save", tests=True, data=dataset)
        if not isinstance(models, Iterable):
            models = [models]
        for model in models:
            n_corrects = 0
            n_try = 10
            instances = learner.get_instances(model=model, n=n_try, indexes=Learning.TEST)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                #if round(prediction_classifier,5) == round(prediction_model_1,5):
                #    n_corrects += 1
                self.assertAlmostEqual(prediction_classifier, prediction_model_1, places=10)
                self.assertAlmostEqual(prediction_classifier, prediction_model_2, places=10)
            #self.assertGreaterEqual(n_corrects, n_try*0.9)

    
    def save_load_continuity(self, dataset, problem_type, model_type, labels_type):
        n_instances = 10
        learner = Learning.LightGBM(dataset, problem_type=problem_type, labels_type=labels_type)
        model = learner.evaluate(splitting_method='hold-out', model_type=model_type, splitting_parameters={"test_size":0.23, "random_state":0})
        instances_initial = learner.get_instances(model=model, n=n_instances, indexes=Learning.TEST)

        Learning.ModelIO.save(model, "try_save")
        del learner
        del model
        learner, model = Learning.ModelIO.load(models_directory="try_save", tests=True, data=dataset)
        instances_load = learner.get_instances(model=model, n=n_instances, indexes=Learning.TEST)

        for i in range(n_instances):
            self.assertEqual(instances_initial[i][1], instances_load[i][1])

if __name__ == '__main__':
    print("Tests: " + TestSaveLoadLightGBM.__name__ + ":")
    unittest.main()
