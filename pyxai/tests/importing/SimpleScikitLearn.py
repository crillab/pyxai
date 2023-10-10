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


    def test_simple_RF_breast_cancer(self):
        model_rf = RandomForestClassifier(random_state=0)
        data = datasets.load_breast_cancer(as_frame=True)
        X = data.data.to_numpy()
        Y = data.target.to_numpy()

        feature_names = data.feature_names
        model_rf.fit(X, Y)

        learner, model = Learning.import_models(model_rf, feature_names)
        instance, prediction = learner.get_instances(dataset=data.frame, model=model, n=1)
        
        explainer = Explainer.initialize(model, instance=instance)

        direct = explainer.direct_reason()
        
        sufficient = explainer.sufficient_reason()
        

    def test_simple_RF_iris(self):
        model_rf = RandomForestClassifier(random_state=0)
        data = datasets.load_iris(as_frame=True)
        X = data.data.to_numpy()
        Y = data.target.to_numpy()

        feature_names = data.feature_names
        model_rf.fit(X, Y)

        learner, model = Learning.import_models(model_rf)
        instance, prediction = learner.get_instances(dataset=data.frame, model=model, n=1)
        
        explainer = Explainer.initialize(model, instance=instance)

        direct = explainer.direct_reason()
        
        sufficient = explainer.sufficient_reason()
        

if __name__ == '__main__':
    print("Tests: " + TestImportSimpleScikitlearn.__name__ + ":")
    unittest.main()
