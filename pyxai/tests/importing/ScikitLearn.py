from pyxai import Tools, Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut

import pandas
import numpy
import functools
import operator
import copy

from pyxai.sources.type import _check_parameter, ModelType

Tools.set_verbose(0)

import unittest


class TestImportScikitlearn(unittest.TestCase):
    #@unittest.skip("reason for skipping")
    def test_import_feature_names(self):
        from sklearn.datasets import load_iris
        iris = load_iris()
        full_data = numpy.concatenate([iris.data, iris.target.reshape(-1,1)], axis=1)
        iris_df = pandas.DataFrame(full_data, columns=iris.feature_names+['label'])
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=0.5,
                                    min_samples_split=0.3, min_weight_fraction_leaf=0.1, criterion='entropy',
                                    random_state=2023)
        
        rf.fit(iris_df.drop(columns=['label']), iris_df['label'])


        learner, model = Learning.ModelIO.import_models(rf, iris_df)
        for f in ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']:
            self.assertIn(f, learner.feature_names)
        

    def test_import_dataset_import_models(self):
        self.do_import_dataset_import_models("tests/datasets/iris.csv", Learning.RF)
        self.do_import_dataset_import_models("tests/datasets/iris.csv", Learning.DT)
        self.do_import_dataset_import_models("tests/datasets/weatherAUS_simplified.csv", Learning.RF)
        self.do_import_dataset_import_models("tests/datasets/weatherAUS_simplified.csv", Learning.DT)
        self.do_import_dataset_import_models("tests/datasets/winequality-red.csv", Learning.RF)
        self.do_import_dataset_import_models("tests/datasets/winequality-red.csv", Learning.DT)

    
    #@unittest.skip("reason for skipping")
    def test_import_dataset_get_instances(self):
        self.do_import_dataset_get_instances("tests/datasets/iris.csv", Learning.RF, Learning.InstancesType.Tabular)
        self.do_import_dataset_get_instances("tests/datasets/iris.csv", Learning.DT, Learning.InstancesType.Tabular)
        self.do_import_dataset_get_instances("tests/datasets/weatherAUS_simplified.csv", 'random-forest', 'tabular')
        self.do_import_dataset_get_instances("tests/datasets/weatherAUS_simplified.csv", 'decision-tree', 'tabular')
        self.do_import_dataset_get_instances("tests/datasets/winequality-red.csv", 'random-forest', 'tabular')
        self.do_import_dataset_get_instances("tests/datasets/winequality-red.csv", 'decision-tree', 'tabular')

    #Warning: we have <0.01% of errors due to precision errors
    def do_import_dataset_import_models(self, dataset, model_type, instances_type=None, labels_type=None):
        _, _, sk_models, _, test_indexes = self.load_dataset(dataset, model_type)

        learner, models = Learning.ModelIO.import_models(sk_models, dataset, instances_type=instances_type, labels_type=labels_type)


        n_tests = 0
        n_errors = 0
        for i, model in enumerate(models):
            instances = learner.get_instances(model=model, n=20, indexes=Learning.TEST, test_indexes=test_indexes[i])
            
            for (instance, prediction_init_model) in instances:
                prediction_pyxai_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_pyxai_model_2 = model.predict_implicant(implicant)
                if (prediction_init_model != prediction_pyxai_model_1):
                    n_errors+=1
                n_tests+=1
        
        
        self.assertLessEqual(n_errors/n_tests, 0.01)
        
                
    #Warning: we have <0.01% of errors due to precision errors
    def do_import_dataset_get_instances(self, dataset, model_type, instances_type=None, labels_type=None):
        _, _, sk_models, _, _ = self.load_dataset(dataset, model_type)

        learner, models = Learning.ModelIO.import_models(sk_models, instances_type=instances_type, labels_type=labels_type)

        n_tests = 0
        n_errors = 0
        for i, model in enumerate(models):
            instances = learner.get_instances(dataset=dataset, model=model, n=20)
            
            for (instance, prediction_init_model) in instances:
                prediction_pyxai_model_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_pyxai_model_2 = model.predict_implicant(implicant)
                if (prediction_init_model != prediction_pyxai_model_1):
                    n_errors+=1
                n_tests+=1
        
        
        self.assertLessEqual(n_errors/n_tests, 0.01)
    

    def load_dataset(self, dataset, model_type):
        data = pandas.read_csv(dataset).copy()
        labels = data[data.columns[-1]]
        # remove the label of each instance
        data = data.drop(columns=[data.columns[-1]])

        results = self.cross_validation(data, labels, model_type, n_trees=5)
        sk_models = [result[0] for result in results]
        train_indexes = [result[1] for result in results]
        test_indexes = [result[2] for result in results]

        return data, labels, sk_models, train_indexes, test_indexes


    def cross_validation(self, X, Y, model_type, n_trees=5, n_forests=2):
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
        for index_train, index_test in loo.split(X, Y, groups=groups):
            # Creation of instances (X) and labels (Y) according to the index of loo.split() 
            # for both training and test set
            x_train = X.iloc[index_train]
            y_train = [Y[x] for x in index_train]
            x_test = X.iloc[index_test]
            y_test = [Y[x] for x in index_test]

            # Training phase
            if _check_parameter(model_type, ModelType, "model_type", "__init__") == ModelType.RandomForest:
                rf = RandomForestClassifier(n_estimators=n_trees, random_state=0)
            else:
                rf = DecisionTreeClassifier(random_state=0)

            rf.fit(x_train, y_train)

            # Get the classifier prediction of the test set  
            y_predict = rf.predict(x_test)

            forests.append((rf, index_train, index_test))
        return forests



if __name__ == '__main__':
    print("Tests: " + TestImportScikitlearn.__name__ + ":")
    unittest.main()
