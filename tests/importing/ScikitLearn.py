from pyxai import Tools, Learning, Explainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut

import pandas
import numpy
import random
import functools
import operator
import copy

Tools.set_verbose(0)

import unittest


class TestImportScikitlearn(unittest.TestCase):
    PRECISION = 3


    def test_import_RF_iris(self):
        self.do_import("tests/iris.csv", Learning.RF)


    def test_import_DT_iris(self):
        self.do_import("tests/iris.csv", Learning.DT)


    def do_import(self, dataset, learner_type):
        data, labels, feature_names = self.load_dataset(dataset)
        results = self.cross_validation(data, labels, learner_type)
        sk_models = [result[0] for result in results]
        training_indexes = [result[1] for result in results]
        test_indexes = [result[2] for result in results]

        learner, models = Learning.import_models(sk_models)

        for i, model in enumerate(models):
            # instances = learner.get_instances(dataset="tests/iris.csv", model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])
            instances = learner.get_instances(dataset=dataset, model=model, n=10)

            for (instance, prediction_classifier) in instances:
                prediction_model_1 = learner.get_label_from_value(model.predict_instance(instance))
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = learner.get_label_from_value(model.predict_implicant(implicant))
                self.assertEqual(prediction_classifier, prediction_model_1)
                self.assertEqual(prediction_classifier, prediction_model_2)


    def load_dataset(self, dataset):
        data = pandas.read_csv(dataset).copy()

        # extract labels
        labels = data[data.columns[-1]]
        labels = numpy.array(labels)

        # remove the label of each instance
        data = data.drop(columns=[data.columns[-1]])

        # extract the feature names
        feature_names = list(data.columns)

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
            if learner_type == Learning.RF:
                rf = RandomForestClassifier(n_estimators=n_trees)
            else:
                rf = DecisionTreeClassifier()
            rf.fit(x_train, y_train)

            # Get the classifier prediction of the test set  
            y_predict = rf.predict(x_test)

            forests.append((rf, index_training, index_test))
        return forests


if __name__ == '__main__':
    print("Tests: " + TestImportScikitlearn.__name__ + ":")
    unittest.main()
