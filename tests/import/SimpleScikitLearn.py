from pyxai import Tools, Learning, Explainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import svm
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

import pandas
import numpy 
import random
import functools
import operator
import copy
Tools.set_verbose(0)

import unittest

class TestImportSimpleScikitlearn(unittest.TestCase):
    PRECISION = 3
    

    def setUp(self):
        print("..|In method:", self._testMethodName)

    def test_simple_RF_breast_cancer(self):
        model_rf = RandomForestClassifier(random_state=0)
        data = datasets.load_breast_cancer(as_frame=True)
        X = data.data.to_numpy()
        Y = data.target.to_numpy()

        feature_names = data.feature_names
        model_rf.fit(X, Y)

        learner, model = Learning.import_models(model_rf, feature_names)
        instance, prediction = learner.get_instances(dataset=data.frame, model=model, n=1)
        print("instance:", instance)
        print("prediction:", prediction)
        
        explainer = Explainer.initialize(model, instance=instance)
        
        direct = explainer.direct_reason()
        print("len direct reason:", len(direct))

        sufficient = explainer.sufficient_reason()
        print("len sufficient reason:", len(sufficient))

        print("to_features:", explainer.to_features(sufficient))

    def test_simple_RF_iris(self):
        model_rf = RandomForestClassifier(random_state=0)
        data = datasets.load_iris(as_frame=True)
        X = data.data.to_numpy()
        Y = data.target.to_numpy()

        feature_names = data.feature_names
        model_rf.fit(X, Y)

        learner, model = Learning.import_models(model_rf)
        instance, prediction = learner.get_instances(dataset=data.frame, model=model, n=1)
        print("instance:", instance)
        print("prediction:", prediction)
        
        explainer = Explainer.initialize(model, instance=instance)
        
        direct = explainer.direct_reason()
        print("len direct reason:", len(direct))

        sufficient = explainer.sufficient_reason()
        print("len sufficient reason:", len(sufficient))

        print("to_features:", explainer.to_features(sufficient))

        

if __name__ == '__main__':
    print("Tests: " + TestImportSimpleScikitlearn.__name__ + ":")
    unittest.main()