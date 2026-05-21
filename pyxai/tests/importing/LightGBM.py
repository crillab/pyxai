from pyxai import Tools, Learning
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm
import pandas
import numpy
import random
import functools
import operator
import copy

from pyxai.sources.type import _check_parameter, ProblemType

Tools.set_verbose(0)

import unittest


class TestImportLightGBM(unittest.TestCase):
    def test_import_dataset_import_models(self):
        self.do_import_dataset_import_models("tests/datasets/dermatology.csv", 'regression')
        self.do_import_dataset_import_models("tests/datasets/winequality-red.csv", 'regression', 'tabular', 'discrete-values')

    def test_import_dataset_get_instances(self):
        self.do_import_dataset_get_instances("tests/datasets/dermatology.csv", 'regression', 'tabular', 'continuous-values')
        self.do_import_dataset_get_instances("tests/datasets/winequality-red.csv", 'regression', 'tabular', 'discrete-values')



    def do_import_dataset_import_models(self, dataset, problem_type, instances_type=None, labels_type=None):
        data, labels, sk_models, train_indexes, test_indexes = self.load_dataset(dataset, problem_type)

        full_dts = pandas.concat([data, labels], axis=1)
        learner, models = Learning.ModelIO.import_models(sk_models, full_dts, instances_type=instances_type, labels_type=labels_type)

        for i, model in enumerate(models):
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])
            self.check_predictions(model, instances, labels_type)


    def do_import_dataset_get_instances(self, dataset, problem_type, instances_type, labels_type):
        data, labels, sk_models, train_indexes, test_indexes = self.load_dataset(dataset, problem_type)

        learner, models = Learning.ModelIO.import_models(sk_models, instances_type=instances_type, labels_type=labels_type)

        for i, model in enumerate(models):
            instances = learner.get_instances(dataset=dataset, model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])
            self.check_predictions(model, instances, labels_type)


    def load_dataset(self, dataset, problem_type):
        data = pandas.read_csv(dataset).copy()
        labels = data[data.columns[-1]]
        # remove the label of each instance
        data = data.drop(columns=[data.columns[-1]])

        results = self.cross_validation(data, labels, problem_type, n_trees=5)
        sk_models = [result[0] for result in results]
        train_indexes = [result[1] for result in results]
        test_indexes = [result[2] for result in results]

        return data, labels, sk_models, train_indexes, test_indexes


    def cross_validation(self, X, Y, problem_type, n_trees=100, n_forests=10):
        n_instance = len(Y)
        quotient = n_instance // n_forests
        remain = n_instance % n_forests

        # Groups creation
        groups = [quotient * [i] for i in range(1, n_forests + 1)]
        groups = functools.reduce(operator.iconcat, groups, [])
        groups += [i for i in range(1, remain + 1)]
        random.shuffle(groups)

        # Variable definition
        loo = LeaveOneGroupOut()
        forests = []
        i = 0
        for index_training, index_test in loo.split(X, Y, groups=groups):
            if i < n_forests:
                i += 1
            # Creation of instances (X) and labels (Y) according to the index of loo.split() 
            # for both training and test set
            x_train = X.iloc[index_training]
            y_train = Y.iloc[index_training]
            x_test = X.iloc[index_test]
            y_test = Y.iloc[index_test]
            # Training phase
            learner = lightgbm.LGBMRegressor(verbose=-1, n_estimators=5, random_state=0)
            learner.fit(x_train, y_train)
            # Get the classifier prediction of the test set  
            y_predict = learner.predict(x_test)

            forests.append((learner, index_training, index_test))
        return forests

    def check_predictions(self, model, instances, labels_type):
        for (instance, prediction_init_model) in instances:
            prediction_pyxai_model_1 = model.predict_instance(instance)
            implicant = model.instance_to_binaries(instance)
            prediction_pyxai_model_2 = model.predict_implicant(implicant)
            if labels_type=='discrete-values':
                prediction_init_model = round(prediction_init_model)
            self.assertAlmostEqual(prediction_init_model, prediction_pyxai_model_1, places=14)
            self.assertAlmostEqual(prediction_init_model, prediction_pyxai_model_2, places=14)



if __name__ == '__main__':
    print("Tests: " + TestImportLightGBM.__name__ + ":")
    unittest.main()
