from pyxai import Tools, Learning, Explainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

import pandas
import numpy 
import random
import functools
import operator
import copy
Tools.set_verbose(1)

import unittest

class TestImportScikitlearn(unittest.TestCase):
    PRECISION = 3
    

    def setUp(self):
        print("..|In method:", self._testMethodName)

    def test_import(self):
        data, labels, feature_names = self.load_dataset("tests/iris.csv")
        forests = self.cross_validation(data, labels)
        learner, models = Learning.import_models(forests)

        for i, model in enumerate(models):
            instances = learner.get_instances(dataset="tests/iris.csv", model=model, n=10)
            for (instance, prediction_classifier) in instances:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_classifier,prediction_model_1)
                self.assertEqual(prediction_classifier,prediction_model_2)

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

    def cross_validation(self, X, Y, n_trees=100, n_forests=10) :
        n_instance = len(Y)
        quotient = n_instance // n_forests
        remain = n_instance % n_forests
        
        # Groups creation
        groups = [quotient*[i] for i in range(1,n_forests+1)]
        groups = functools.reduce(operator.iconcat, groups, [])
        groups += [i for i in range(1,remain+1)]
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
            rf = RandomForestClassifier(n_estimators=n_trees)
            rf.fit(x_train, y_train)
            
            # Get the classifier prediction of the test set  
            y_predict = rf.predict(x_test)
            
            forests.append(rf)
        return forests

if __name__ == '__main__':
    print("Tests: " + TestImportScikitlearn.__name__ + ":")
    unittest.main()