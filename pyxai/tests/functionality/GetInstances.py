from pyxai import Tools, Learning
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
import lightgbm
import pandas
import numpy 
import random
import functools
import operator
import copy

Tools.set_verbose(0)

# Needs to add tests without n_selected
# Needs to test is_correct

import unittest
class TestGetInstances(unittest.TestCase):
    #@unittest.skip("reason for skipping")
    def test_subset_column(self):
        data_iris = self.add_subset_column("tests/datasets/iris.csv")
        data_dermato = self.add_subset_column("tests/datasets/dermatology.csv")
        data_winequality = self.add_subset_column("tests/datasets/winequality-red.csv")
        
        get_instances_parameters ={"n":1, "indexes":'test'}
        self.main_subset_type_assertions(data_iris, "Scikitlearn", "classification", "predefined", "random-forest", {}, {'max_depth':6, 'random_state':0}, get_instances_parameters)
        
        get_instances_parameters ={"subset_true_classes":['Iris-versicolor'], "indexes":'train'}
        self.main_subset_type_assertions(data_iris, "Scikitlearn", "classification", "hold-out", "random-forest", {'test_size':0.2, 'random_state':0}, {'max_depth':5, 'random_state':0}, get_instances_parameters)

        get_instances_parameters ={"n":1, "indexes":'test', "is_correct":True}
        self.main_subset_type_assertions(data_iris, "Scikitlearn", "classification", "leave-one-group-out", "random-forest", {'random_state':0}, {'max_depth':4, 'random_state':0}, get_instances_parameters)
        
        get_instances_parameters ={"n":2, "indexes":'train', "is_correct":True}
        self.main_subset_type_assertions(data_iris, "Scikitlearn", "classification", "k-folds", "random-forest", {'shuffle':True, 'random_state':0}, {'max_depth':7, 'random_state':0}, get_instances_parameters)

        get_instances_parameters ={"indexes":'train', "subset_true_classes":[0]}
        self.main_subset_type_assertions(data_dermato, "Xgboost", "classification", "hold-out", "boosted-tree", {'test_size':0.2, 'random_state':0}, {'max_depth':2, 'random_state':0}, get_instances_parameters)

        get_instances_parameters ={"n":1, "indexes":'test'}
        self.main_subset_type_assertions(data_dermato, "Xgboost", "classification", "predefined", "boosted-tree", {}, {'max_depth':2, 'random_state':0}, get_instances_parameters)

        get_instances_parameters ={"n":1, "indexes":'test'}
        self.main_subset_type_assertions(data_dermato, "Xgboost", "regression", "leave-one-group-out", "boosted-tree", {'random_state':0}, {'max_depth':2, 'random_state':0}, get_instances_parameters)
        
        get_instances_parameters ={"n":2, "indexes":'train'}
        self.main_subset_type_assertions(data_dermato, "Xgboost", "regression", "k-folds", "boosted-tree", {}, {'max_depth':3, 'random_state':0}, get_instances_parameters)
        
        get_instances_parameters ={"indexes":'train'}
        self.main_subset_type_assertions(data_winequality, "LightGBM", "regression", "hold-out", "boosted-tree", {'test_size':0.2, 'random_state':0}, {'max_depth':3, 'random_state':0}, get_instances_parameters)

        get_instances_parameters ={"n":1, "indexes":'test'}
        self.main_subset_type_assertions(data_winequality, "LightGBM", "regression", "leave-one-group-out", "boosted-tree", {'random_state':0}, {'max_depth':2, 'random_state':0}, get_instances_parameters)
        
        get_instances_parameters ={"n":2, "indexes":'train'}
        self.main_subset_type_assertions(data_winequality, "LightGBM", "regression", "k-folds", "boosted-tree", {}, {'max_depth':2, 'random_state':0}, get_instances_parameters)

        get_instances_parameters ={"n":1, "indexes":'test'}
        self.main_subset_type_assertions(data_winequality, "LightGBM", "regression", "predefined", "boosted-tree", {}, {'max_depth':2, 'random_state':0}, get_instances_parameters)
        

    #@unittest.skip("reason for skipping")
    def test_get_instances_simple(self):
        self.main_simple("tests/datasets/dermatology.csv", "LightGBM", Learning.REGRESSION, Learning.K_FOLDS, Learning.BT, {'shuffle':True, 'random_state':0}, {'learning_rate':0.3, 'n_estimators':5}, {"n":10})
        self.main_simple("tests/datasets/winequality-red.csv", "LightGBM", Learning.REGRESSION, Learning.K_FOLDS, Learning.BT, {}, {'learning_rate':0.3, 'n_estimators':5}, {"n":10})


    #@unittest.skip("reason for skipping")
    def test_get_instances_import(self):
        self.main_import("tests/datasets/iris.csv", RandomForestClassifier(n_estimators=5, random_state=0), {"n":10, "indexes":Learning.TEST})
        self.main_import("tests/datasets/iris.csv", RandomForestClassifier(n_estimators=5, random_state=0), {"n":None, "indexes":Learning.ALL})
        self.main_import("tests/datasets/compas.csv", RandomForestClassifier(n_estimators=5, random_state=0), {"n":10, "indexes":Learning.TRAIN})
        self.main_import("tests/datasets/compas.csv", RandomForestClassifier(n_estimators=5, random_state=0), {"n":None, "indexes":Learning.TRAIN_IN_PRIORITY})
        self.main_import("tests/datasets/weatherAUS_simplified.csv", RandomForestClassifier(n_estimators=5, random_state=0), {"n":10, "indexes":"train-in-priority"})
        self.main_import("tests/datasets/weatherAUS_simplified.csv", RandomForestClassifier(n_estimators=5, random_state=0), {"n":None, "indexes":"train"})


    #@unittest.skip("reason for skipping")
    def test_get_instances_is_correct(self):
        get_instances_parameters ={"n":5, "indexes":'test', "is_correct":True}
        self.main_get_instances_parameters("tests/datasets/weatherAUS_simplified.csv", "Scikitlearn", "classification", "hold-out", "decision-tree", {'random_state':0}, {'max_depth':3}, get_instances_parameters)

        get_instances_parameters ={"n":5, "indexes":'test', "is_correct":False}
        self.main_get_instances_parameters("tests/datasets/weatherAUS_simplified.csv", "Scikitlearn", "classification", "hold-out", "decision-tree", {'random_state':0}, {'max_depth':3}, get_instances_parameters)

        get_instances_parameters ={"n":5, "subset_true_classes":[0]}
        self.main_get_instances_parameters("tests/datasets/compas.csv", "Scikitlearn", "classification", "hold-out", "decision-tree", {'random_state':0}, {'max_depth':3}, get_instances_parameters)

        get_instances_parameters ={"n":5, "subset_true_classes":["Yes"]}
        self.main_get_instances_parameters("tests/datasets/weatherAUS_simplified.csv", "Scikitlearn", Learning.CLASSIFICATION, "hold-out", "decision-tree", {}, {'max_depth':3}, get_instances_parameters)

        get_instances_parameters ={"indexes":"test", "subset_predicted_classes":['Iris-versicolor', 'Iris-virginica']}
        self.main_get_instances_parameters("tests/datasets/iris.csv", "Scikitlearn", "classification", "hold-out", "decision-tree", {'test_size':0.5}, {'max_depth':3}, get_instances_parameters)

        get_instances_parameters ={"n":10, "indexes":"test", "is_correct":False, "subset_true_classes":['Iris-versicolor', 'Iris-virginica'], "subset_predicted_classes":['Iris-versicolor', 'Iris-virginica']}
        self.main_get_instances_parameters("tests/datasets/iris.csv", "Scikitlearn", "classification", "hold-out", "decision-tree", {'random_state':0, 'test_size':0.5}, {'max_depth':3}, get_instances_parameters)


    #@unittest.skip("reason for skipping")
    def test_get_instances_seed(self):
        dataset = "tests/datasets/winequality-red.csv"
        n_selected = 5
        learner = Learning.Scikitlearn(dataset, problem_type='regression')
        model = learner.evaluate(splitting_method='hold-out', model_type='linear-ridge', splitting_parameters={'random_state':0}, model_parameters={})
        instances = learner.get_instances(model, indexes='train-in-priority', n=n_selected, seed=72)
        instance = [instance for (instance, _) in instances]
        
        learner_bis = Learning.Scikitlearn(dataset, problem_type='regression')
        model_bis = learner_bis.evaluate(splitting_method='hold-out', model_type='linear-ridge', splitting_parameters={'random_state':0}, model_parameters={})
        instances_bis = learner_bis.get_instances(model_bis, indexes='train-in-priority', n=n_selected, seed=72) 
        instance_bis = [instance for (instance, _) in instances_bis]

        for j in range(n_selected):
            self.assertEqual(len(instance[j]),len(instance_bis[j]))
            for k in range(len(instance[j])):
                self.assertEqual(instance[j].iloc[k],instance_bis[j].iloc[k])


    #@unittest.skip("reason for skipping")
    def test_get_instances_all_instances(self):
        dataset = "tests/datasets/dermatology.csv"
        reference_df = pandas.read_csv(dataset)
        
        learner = Learning.Scikitlearn(dataset, problem_type='classification')
        model = learner.evaluate(splitting_method='hold-out', model_type='decision-tree', splitting_parameters={'test_size':0.2}, model_parameters={'max_depth':5})
        instances = learner.get_instances(model = model, seed=None)
        self.assertEqual(len(instances), len(reference_df))
        feature_33 = [instance.iloc[33] for (instance, _) in instances]
        instance_vc = pandas.Series(feature_33).value_counts()
        reference_vc = reference_df.iloc[:,33].value_counts()
        for idx in instance_vc.index:
            self.assertEqual(instance_vc[idx], reference_vc[idx])



    def add_subset_column(self, path):
        data = pandas.read_csv(path)
        data.insert(0, 'subset', 'train')
        for i in random.sample(data.index.to_list(), k=len(data)//5):
            data.loc[i, 'subset'] = 'test'
        return data

    def main_subset_type_assertions(self, data, learner_name, problem_type, splitting_method, model_type, splitting_parameters, model_parameters, get_instances_parameters):
        learner = self.get_learner(data, learner_name, problem_type)
        model = learner.evaluate(splitting_method=splitting_method, model_type=model_type, splitting_parameters=splitting_parameters, model_parameters=model_parameters)
        if splitting_method in ["leave-one-group-out", "k-folds"]:
            model = model[0]
        instances = learner.get_instances(model, **get_instances_parameters)

        if "n" in get_instances_parameters.keys() and get_instances_parameters["n"]==1:
            self.assertEqual(type(instances), tuple)
            self.assertEqual(type(instances[0]), pandas.Series)
        else:
            self.assertEqual(type(instances), list)
            self.assertEqual(type(instances[0]), tuple)
            self.assertEqual(type(instances[0][0]), pandas.Series)


    
    def main_simple(self, data, learner_name, problem_type, splitting_method, model_type, splitting_parameters, model_parameters, get_instances_parameters):
        learner = self.get_learner(data, learner_name, problem_type)

        models = learner.evaluate(splitting_method=splitting_method, model_type=model_type, splitting_parameters=splitting_parameters, model_parameters=model_parameters)
        for i, model in enumerate(models):
            data = learner.instances_dataframe
            instances = learner.get_instances(model=model, **get_instances_parameters)
            instance = [instance for (instance, _) in instances]
            true_instance = [data.iloc[j] for j in range(get_instances_parameters["n"])]
            
            for j in range(get_instances_parameters["n"]):
                self.assertEqual(len(true_instance[j]),len(instance[j]))
                for k in range(len(true_instance[j])):
                    self.assertEqual(true_instance[j].iloc[k],instance[j].iloc[k])



    def do_import(self, dataset, imported_learner, n_forests):
        data, labels, feature_names = self.load_dataset(dataset)
        results = self.cross_validation(data, labels, imported_learner, n_forests)
        sk_models = [result[0] for result in results]
        train_indexes = [result[1] for result in results]
        test_indexes = [result[2] for result in results]
        
        learner, models = Learning.ModelIO.import_models(sk_models, pandas.concat([data, labels], axis=1))
        return data, labels, learner, models, train_indexes, test_indexes

    def load_dataset(self, dataset):
        data = pandas.read_csv(dataset).copy()

        # extract labels
        labels = data.iloc[:,-1]
        #labels = numpy.array(labels)

        # remove the label of each instance
        data = data.drop(columns=[data.columns[-1]])
        
        # extract the feature names
        feature_names = list(data.columns)
        return data, labels, feature_names

    def cross_validation(self, X, Y, imported_learner, n_forests):
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
        for index_train, index_test in loo.split(X, Y, groups=groups):
            if i < n_forests:
                i += 1
            # Creation of instances (X) and labels (Y) according to the index of loo.split() for both train and test set
            x_train = X.iloc[index_train]
            y_train = [Y[x] for x in index_train]
            x_test = X.iloc[index_test]
            y_test = [Y[x] for x in index_test]
            
            # Training phase
            imported_learner.fit(x_train, y_train)
            # Get the classifier prediction of the test set  
            y_predict = imported_learner.predict(x_test)
            
            forests.append((imported_learner, index_train, index_test))
        return forests

    def main_import(self, data_path, imported_learner, get_instances_parameters):
        data, _, learner, models, train_indexes, test_indexes = self.do_import(data_path, imported_learner, 5)
        for i, model in enumerate(models):
            n_selected = get_instances_parameters["n"]
            instances = learner.get_instances(model=model, n=n_selected, indexes=get_instances_parameters["indexes"], train_indexes=train_indexes[i], test_indexes=test_indexes[i])
            if n_selected is not None:
                self.assertLessEqual(len(instances), n_selected)
            else:
                n_selected = len(data)
            instance = [instance for (instance, _) in instances]
            if get_instances_parameters["indexes"] == Learning.TEST:
                selection = test_indexes[i][:n_selected]
            elif get_instances_parameters["indexes"] == Learning.TRAIN:
                selection = train_indexes[i][:n_selected]
            elif get_instances_parameters["indexes"] == Learning.TRAIN_IN_PRIORITY:
                if n_selected<len(train_indexes[i]):
                    selection = train_indexes[i][:n_selected]
                else:
                    selection = numpy.concatenate((train_indexes[i], test_indexes[i][:n_selected-len(train_indexes)]))
            else:
                selection = range(n_selected)
            true_instance = [data.values[j] for j in selection]

            self.assertEqual(len(true_instance), len(instances))
            for j in range(len(true_instance)):
                self.assertEqual(len(true_instance[j]),len(instance[j]))
                for k in range(len(true_instance[j])):
                    self.assertEqual(true_instance[j][k],instance[j].iloc[k])


    def main_get_instances_parameters(self, data, learner_name, problem_type, splitting_method, model_type, splitting_parameters, model_parameters, get_instances_parameters):
        learner = self.get_learner(data, learner_name, problem_type)
        model = learner.evaluate(splitting_method=splitting_method, model_type=model_type, splitting_parameters=splitting_parameters, model_parameters=model_parameters)
        if splitting_method in ["leave-one-group-out", "k-folds"]:
            model = model[0]
        instances = learner.get_instances(model, **get_instances_parameters)
        if "n" in get_instances_parameters.keys():
            self.assertLessEqual(len(instances), get_instances_parameters["n"])
        for instance, prediction in instances:
            instance_true = learner.labels_dataframe.label.loc[instance.name]
            if "is_correct" in get_instances_parameters.keys() and get_instances_parameters["is_correct"]==True:
                self.assertEqual(instance_true, prediction)
            if "is_correct" in get_instances_parameters.keys() and get_instances_parameters["is_correct"]==False:
                self.assertNotEqual(instance_true, prediction)
            if "subset_true_classes" in get_instances_parameters.keys():
                self.assertIn(instance_true, get_instances_parameters["subset_true_classes"])
            if "subset_predicted_classes" in get_instances_parameters.keys():
                self.assertIn(prediction, get_instances_parameters["subset_predicted_classes"])
            


    def get_learner(self, data, learner_name, problem_type):
        if learner_name=="Scikitlearn":
            learner = Learning.Scikitlearn(data, problem_type=problem_type)
        elif learner_name=="Xgboost":
            learner = Learning.Xgboost(data, problem_type=problem_type)
        elif learner_name=="LightGBM":
            learner = Learning.LightGBM(data, problem_type=problem_type)
        else:
            raise ValueError("Unrecognised learner_name")
        return learner

if __name__ == '__main__':
    print("Tests: " + TestGetInstances.__name__ + ":")
    unittest.main()