from pyxai import Tools, Learning
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
import xgboost
import pandas
import numpy
import random
import functools
import operator
import copy

from pyxai.sources.type import _check_parameter, ProblemType

Tools.set_verbose(0)

import unittest


class TestImportXGBoost(unittest.TestCase):
    PRECISION = 4
    N_TESTS_PER_DATASET = 10 #Ok with 100 but 10 to do the tests faster

    def test_import_data_partial(self):
        # Special dataset creation
        dataset = pandas.read_csv("tests/datasets/weatherAUS_simplified.csv", skiprows=range(1,1001), nrows=1000)
        data = dataset.copy()
        data.insert(0, 'subset', 'train')
        for i in random.sample(data.index.to_list(), k=len(data)//5):
            data.loc[i, 'subset'] = 'test'
            
        labels = data[data.columns[-1]] # Test case with series labels
        data = data.drop(columns=[data.columns[-1]])
        feature_names = list(data.columns)
        
        model = xgboost.XGBClassifier()

        # Manual part
        index_train = data[data.subset=='train'].index
        index_test = data[data.subset=='train'].index
        data = data.drop(columns=['subset'])
        x_train = data.iloc[index_train]
        y_train = [labels[x] for x in index_train]
        x_test = data.iloc[index_test]
        y_test = [labels[x] for x in index_test]
        
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        model.fit(x_train, y_train)
        result = model.predict(x_test)
        result = label_encoder.inverse_transform(result)

        # Pyxai part
        learner, pyxai_model = Learning.ModelIO.import_models(model, instances_type='tabular')
        instances = learner.get_instances(pyxai_model, dataset=dataset, n=10, indexes='test', test_indexes=index_test)
        for (instance, prediction_classifier) in instances:
            prediction_pyxai = pyxai_model.predict_instance(instance)
            implicant = pyxai_model.instance_to_binaries(instance)
            prediction_implicant = pyxai_model.predict_implicant(implicant)
            self.assertAlmostEqual(prediction_classifier, prediction_pyxai, TestImportXGBoost.PRECISION)
            self.assertAlmostEqual(prediction_classifier, prediction_implicant, TestImportXGBoost.PRECISION)

    
    #@unittest.skip("reason for skipping")
    def test_import_dataset_import_models(self):
        self.do_import_dataset_import_models("tests/datasets/iris.csv", Learning.CLASSIFICATION)
        self.do_import_dataset_import_models("tests/datasets/dermatology.csv", Learning.CLASSIFICATION)
        self.do_import_dataset_import_models("tests/datasets/winequality-red.csv", Learning.CLASSIFICATION)
        self.do_import_dataset_import_models("tests/datasets/weatherAUS_simplified.csv", 'classification')
        self.do_import_dataset_import_models("tests/datasets/dermatology.csv", 'regression')
        self.do_import_dataset_import_models("tests/datasets/winequality-red.csv", Learning.REGRESSION)

    #@unittest.skip("reason for skipping")
    def test_import_dataset_get_instances(self):
        self.do_import_dataset_get_instances("tests/datasets/iris.csv", Learning.CLASSIFICATION, Learning.InstancesType.Tabular)
        self.do_import_dataset_get_instances("tests/datasets/dermatology.csv", Learning.CLASSIFICATION, Learning.InstancesType.Tabular)
        self.do_import_dataset_get_instances("tests/datasets/winequality-red.csv", Learning.CLASSIFICATION, Learning.InstancesType.Tabular)
        self.do_import_dataset_get_instances("tests/datasets/weatherAUS_simplified.csv", 'classification', 'tabular')
        self.do_import_dataset_get_instances("tests/datasets/dermatology.csv", 'regression', 'tabular')
        self.do_import_dataset_get_instances("tests/datasets/winequality-red.csv", Learning.REGRESSION, Learning.InstancesType.Tabular)


    def do_import_dataset_import_models(self, dataset, problem_type, instances_type=None):
        data, labels, sk_models, _, test_indexes = self.load_dataset(dataset, problem_type)

        learner, models = Learning.ModelIO.import_models(sk_models, dataset, instances_type=instances_type)
        for i, model in enumerate(models):
            instances = learner.get_instances(model=model, n=TestImportXGBoost.N_TESTS_PER_DATASET, indexes=Learning.TEST, test_indexes=test_indexes[i])
            self.check_predictions(model, instances, problem_type)


    def do_import_dataset_get_instances(self, dataset, problem_type, instances_type=None):
        _, _, sk_models, _, test_indexes = self.load_dataset(dataset, problem_type)

        learner, models = Learning.ModelIO.import_models(sk_models, instances_type=instances_type)
        for i, model in enumerate(models):
            instances = learner.get_instances(dataset=dataset, model=model, n=TestImportXGBoost.N_TESTS_PER_DATASET, indexes=Learning.TEST, test_indexes=test_indexes[i])
            self.check_predictions(model, instances, problem_type)

    def load_dataset(self, dataset, problem_type):
        data = pandas.read_csv(dataset)
        labels = data[data.columns[-1]]
        # remove the label of each instance
        data = data.drop(columns=[data.columns[-1]])

        results = self.cross_validation(data, labels, problem_type, n_trees=5)
        sk_models = [result[0] for result in results]
        train_indexes = [result[1] for result in results]
        test_indexes = [result[2] for result in results]

        return data, labels, sk_models, train_indexes, test_indexes


    def cross_validation(self, X, Y, problem_type, n_trees=100, n_forests=4):
        n_instance = len(Y)
        quotient = n_instance // n_forests
        remain = n_instance % n_forests

        # Groups creation
        groups = [quotient * [i] for i in range(1, n_forests + 1)]
        groups = functools.reduce(operator.iconcat, groups, [])
        groups += [i for i in range(1, remain + 1)]
        numpy.random.seed(0)
        numpy.random.shuffle(groups)

        # Variable definition
        loo = LeaveOneGroupOut()
        forests = []
        i = 0
        for index_train, index_test in loo.split(X, Y, groups=groups):
            if i < n_forests:
                i += 1
            # Creation of instances (X) and labels (Y) according to the index of loo.split() 
            # for both training and test set
            x_train = X.iloc[index_train]
            y_train = [Y[x] for x in index_train]
            x_test = X.iloc[index_test]
            y_test = [Y[x] for x in index_test]

            # Training phase
            if _check_parameter(problem_type, ProblemType, "problem_type", "__init__") == 'classification':
                model = xgboost.XGBClassifier(random_state=0)
            elif _check_parameter(problem_type, ProblemType, "problem_type", "__init__") == 'regression':
                model = xgboost.XGBRegressor(n_estimators=5, random_state=0) 

            if problem_type==ProblemType.Classification:
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
            model.fit(x_train, y_train)
            result = model.predict(x_test)
            if problem_type==ProblemType.Classification:
                result = label_encoder.inverse_transform(result)
            forests.append((model, index_train, index_test))
        return forests



    def check_predictions(self, model, instances, problem_type):
        for (instance, prediction_classifier) in instances:
            if _check_parameter(problem_type, ProblemType, "problem_type", "__init__") == 'regression':
                prediction_classifier = round(float(prediction_classifier), TestImportXGBoost.PRECISION)
                prediction_model_1 = round(model.predict_instance(instance), TestImportXGBoost.PRECISION)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = round(model.predict_implicant(implicant), TestImportXGBoost.PRECISION)
            else:
                prediction_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_model_2 = model.predict_implicant(implicant)
            self.assertEqual(prediction_classifier, prediction_model_1)
            self.assertEqual(prediction_classifier, prediction_model_2)



if __name__ == '__main__':
    print("Tests: " + TestImportXGBoost.__name__ + ":")
    unittest.main()
