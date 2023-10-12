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

class TestGetInstances(unittest.TestCase):
    
    def test_get_instances_simple_1(self):
        dataset = "tests/dermatology.csv"
        learner = Learning.LightGBM(dataset, learner_type=Learning.REGRESSION)
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2, learning_rate=0.3, n_estimators=5)
        
        for i, model in enumerate(models):
            #instances = learner.get_instances(dataset="tests/iris.csv", model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])
            test_indexes = learner.learner_information[i].test_index
            data = learner.data
            instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
            instance = [instance for (instance, _) in instances]
            true_instance = [data[test_indexes[j]] for j in range(10)]
            
            for j in range(10):
                self.assertEqual(len(true_instance[j]),len(instance[j]))
                for k in range(len(true_instance[j])):
                    self.assertEqual(true_instance[j][k],instance[j][k])
    
    def test_get_instances_simple_2(self):
        dataset = "tests/dermatology.csv"
        learner = Learning.LightGBM(dataset, learner_type=Learning.REGRESSION)
        models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2, learning_rate=0.3, n_estimators=5)
        
        for i, model in enumerate(models):
            #instances = learner.get_instances(dataset="tests/iris.csv", model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])
            data = learner.data
            instances = learner.get_instances(n=10)
            instance = [instance for (instance, _) in instances]
            true_instance = [data[j] for j in range(10)]
            
            for j in range(10):
                self.assertEqual(len(true_instance[j]),len(instance[j]))
                for k in range(len(true_instance[j])):
                    self.assertEqual(true_instance[j][k],instance[j][k])
    
    def test_get_instances_import_1(self):
        dataset = "tests/iris.csv"
        data, labels, learner, models, training_indexes, test_indexes = self.do_import(dataset, Learning.REGRESSION)
        for i, model in enumerate(models):
            #instances = learner.get_instances(dataset="tests/iris.csv", model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])
            instances = learner.get_instances(dataset=dataset, model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])
            instance = [instance for (instance, _) in instances]
            true_instance = [data[test_indexes[i][j]] for j in range(10)]
            
            for j in range(10):
                self.assertEqual(len(true_instance[j]),len(instance[j]))
                for k in range(len(true_instance[j])):
                    self.assertEqual(true_instance[j][k],instance[j][k])
                    
    def test_get_instances_import_2(self):
        dataset = "tests/iris.csv"
        data, labels, learner, models, training_indexes, test_indexes = self.do_import(dataset, Learning.REGRESSION)
        for i, model in enumerate(models):
            #instances = learner.get_instances(dataset="tests/iris.csv", model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])
            instances = learner.get_instances(dataset=dataset, model=model, n=10, indexes=Learning.TRAINING, training_indexes=training_indexes[i])
            instance = [instance for (instance, _) in instances]
            true_instance = [data[training_indexes[i][j]] for j in range(10)]
            
            for j in range(10):
                self.assertEqual(len(true_instance[j]),len(instance[j]))
                for k in range(len(true_instance[j])):
                    self.assertEqual(true_instance[j][k],instance[j][k])
    
    def test_get_instances_import_3(self):
        dataset = "tests/iris.csv"
        data, labels, learner, models, training_indexes, test_indexes = self.do_import(dataset, Learning.REGRESSION)
        for i, model in enumerate(models):
            #instances = learner.get_instances(dataset="tests/iris.csv", model=model, n=10, indexes=Learning.TEST, test_indexes=test_indexes[i])
            instances = learner.get_instances(dataset=dataset, n=100)
            instance = [instance for (instance, _) in instances]
            true_instance = [data[j] for j in range(10)]
            
            for j in range(10):
                self.assertEqual(len(true_instance[j]),len(instance[j]))
                for k in range(len(true_instance[j])):
                    self.assertEqual(true_instance[j][k],instance[j][k])
                

    def do_import(self, dataset, learner_type):
        data, labels, feature_names = self.load_dataset(dataset, learner_type)
        results = self.cross_validation(data, labels, learner_type, n_trees=5)
        sk_models = [result[0] for result in results]
        training_indexes = [result[1] for result in results]
        test_indexes = [result[2] for result in results]
        
        learner, models = Learning.import_models(sk_models)
        return data, labels, learner, models, training_indexes, test_indexes

    def load_dataset(self, dataset, learner_type):
        data = pandas.read_csv(dataset).copy()

        # extract labels
        labels = data[data.columns[-1]]
        labels = numpy.array(labels)

        # remove the label of each instance
        data = data.drop(columns=[data.columns[-1]])
        
        # extract the feature names
        feature_names = list(data.columns)
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)

        return data.values, labels, feature_names

    def cross_validation(self, X, Y, learner_type, n_trees=100, n_forests=10) :
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
            learner = lightgbm.LGBMRegressor(verbose=-1, n_estimators=5, random_state=0)
            learner.fit(x_train, y_train)
            # Get the classifier prediction of the test set  
            y_predict = learner.predict(x_test)
            
            forests.append((learner, index_training, index_test))
        return forests

if __name__ == '__main__':
    print("Tests: " + TestGetInstances.__name__ + ":")
    unittest.main()