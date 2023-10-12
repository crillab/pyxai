from pyxai import Tools, Learning, Explainer
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm
import pandas
import numpy
import random
import functools
import operator
import copy

Tools.set_verbose(0)

import unittest


class TestImportLightGBM(unittest.TestCase):


    def test_import_BT_regression_dermatology(self):
        self.do_import("tests/dermatology.csv", Learning.REGRESSION)


    def test_import_BT_regression_winequality(self):
        self.do_import("tests/winequality-red.csv", Learning.REGRESSION)


    def do_import(self, dataset, learner_type):
        data, labels, feature_names = self.load_dataset(dataset, learner_type)
        results = self.cross_validation(data, labels, learner_type, n_trees=5)
        sk_models = [result[0] for result in results]
        training_indexes = [result[1] for result in results]
        test_indexes = [result[2] for result in results]

        learner, models = Learning.import_models(sk_models)

        for i, model in enumerate(models):
            # instances = learner.get_instances(dataset="tests/iris.csv", model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])
            instances = learner.get_instances(dataset=dataset, model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])

            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)


    def load_dataset(self, dataset, learner_type):
        data = pandas.read_csv(dataset).copy()

        # extract labels
        labels = data[data.columns[-1]]
        labels = numpy.array(labels)

        # remove the label of each instance
        data = data.drop(columns=[data.columns[-1]])

        # extract the feature names
        feature_names = list(data.columns)
        if learner_type == Learning.CLASSIFICATION:
            le = preprocessing.LabelEncoder()
            le.fit(labels)
            labels = le.transform(labels)

        return data.values, labels, feature_names


    def cross_validation(self, X, Y, learner_type, n_trees=100, n_forests=10):
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
            x_train = [X[x] for x in index_training]
            y_train = [Y[x] for x in index_training]
            x_test = [X[x] for x in index_test]
            y_test = [Y[x] for x in index_test]

            # Training phase
            learner = lightgbm.LGBMRegressor(verbose=-1, n_estimators=5, random_state=0)
            learner.fit(x_train, y_train)
            # Get the classifier prediction of the test set  
            y_predict = learner.predict(x_test)

            forests.append((learner, index_training, index_test))
        return forests


if __name__ == '__main__':
    print("Tests: " + TestImportLightGBM.__name__ + ":")
    unittest.main()
