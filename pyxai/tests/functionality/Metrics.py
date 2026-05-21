from pyxai import Learning, Tools

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut
import unittest
import pandas
import numpy
import random
import functools
import operator

Tools.set_verbose(0)

class TestMetrics(unittest.TestCase):
    PRECISION = 1

    def test_binary_classification(self):
        labels = [0,0,0,0,0,0,0,0,0,0]
        predictions = [0,0,0,0,0,0,0,0,0,0]
        metrics = Tools.Metric.compute_metrics_binary_classification(labels, predictions)
        self.assertEqual(metrics["accuracy"], 100.0)
        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1_score"], 0)
        self.assertEqual(metrics["specificity"], 100.0)
        self.assertEqual(metrics["true_positive"], 0)
        self.assertEqual(metrics["true_negative"], 10)
        self.assertEqual(metrics["false_positive"], 0)
        self.assertEqual(metrics["false_negative"], 0)
        
        labels = [1,1,1,1,1,1,1,1,1,1]
        predictions = [1,1,1,1,1,1,1,1,1,1]
        metrics = Tools.Metric.compute_metrics_binary_classification(labels, predictions)
        self.assertEqual(metrics["accuracy"], 100.0)
        self.assertEqual(metrics["precision"], 100.0)
        self.assertEqual(metrics["recall"], 100.0)
        self.assertEqual(metrics["f1_score"], 100.0)
        self.assertEqual(metrics["specificity"], 0)
        self.assertEqual(metrics["true_positive"], 10)
        self.assertEqual(metrics["true_negative"], 0)
        self.assertEqual(metrics["false_positive"], 0)
        self.assertEqual(metrics["false_negative"], 0)

        labels = [0,0,0,0,0,0,0,0,0,0]
        predictions = [1,1,1,1,1,1,1,1,1,1]
        metrics = Tools.Metric.compute_metrics_binary_classification(labels, predictions)
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1_score"], 0)
        self.assertEqual(metrics["specificity"], 0)
        self.assertEqual(metrics["true_positive"], 0)
        self.assertEqual(metrics["true_negative"], 0)
        self.assertEqual(metrics["false_positive"], 10)
        self.assertEqual(metrics["false_negative"], 0)
        
        labels = [1,1,1,1,1,1,1,1,1,1]
        predictions = [0,0,0,0,0,0,0,0,0,0]
        metrics = Tools.Metric.compute_metrics_binary_classification(labels, predictions)
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1_score"], 0)
        self.assertEqual(metrics["specificity"], 0)
        self.assertEqual(metrics["true_positive"], 0)
        self.assertEqual(metrics["true_negative"], 0)
        self.assertEqual(metrics["false_positive"], 0)
        self.assertEqual(metrics["false_negative"], 10)

        labels = [1,1,1,1,1,0,0,0,0,0]
        predictions = [1,1,1,1,1,0,0,0,0,0]
        metrics = Tools.Metric.compute_metrics_binary_classification(labels, predictions)
        self.assertEqual(metrics["accuracy"], 100.0)
        self.assertEqual(metrics["precision"], 100.0)
        self.assertEqual(metrics["recall"], 100.0)
        self.assertEqual(metrics["f1_score"], 100.0)
        self.assertEqual(metrics["specificity"], 100.0)
        self.assertEqual(metrics["true_positive"], 5)
        self.assertEqual(metrics["true_negative"], 5)
        self.assertEqual(metrics["false_positive"], 0)
        self.assertEqual(metrics["false_negative"], 0)

        labels = [1,1,0,0,0,0,1,1]
        predictions = [1,1,0,0,1,1,0,0]
        metrics = Tools.Metric.compute_metrics_binary_classification(labels, predictions)
        self.assertEqual(metrics["accuracy"], 50.0)
        self.assertEqual(metrics["precision"], 50.0)
        self.assertEqual(metrics["recall"], 50.0)
        self.assertEqual(metrics["f1_score"], 50.0)
        self.assertEqual(metrics["specificity"], 50.0)
        self.assertEqual(metrics["true_positive"], 2)
        self.assertEqual(metrics["true_negative"], 2)
        self.assertEqual(metrics["false_positive"], 2)
        self.assertEqual(metrics["false_negative"], 2)

        labels = pandas.Series(['yes', 'no', 'yes', 'yes'])
        predictions = pandas.Series(['yes', 'yes', 'no', 'yes'])
        metrics = Tools.Metric.compute_metrics_binary_classification(labels, predictions)
        self.assertEqual(metrics["accuracy"], 50.0)

        labels = pandas.Series(['yes', 'no', 'yes', 'maybe', 'yes', 'maybe'])
        predictions = pandas.Series(['yes', 'yes', 'no', 'yes', 'yes', 'maybe'])
        metrics = Tools.Metric.compute_metrics_multi_classification(labels, predictions)
        self.assertAlmostEqual(metrics["micro_averaging_accuracy"], 200/3)
        self.assertAlmostEqual(metrics["micro_averaging_precision"], 50.0)
        self.assertAlmostEqual(metrics["micro_averaging_recall"], 50.0)
        self.assertAlmostEqual(metrics["macro_averaging_accuracy"], 200/3)
        self.assertAlmostEqual(metrics["macro_averaging_precision"], 50.0)
        self.assertAlmostEqual(metrics["macro_averaging_recall"], 700/18)
        self.assertEqual(metrics["true_positives"]['maybe'], 1)
        self.assertEqual(metrics["true_negatives"]['maybe'], 4)
        self.assertEqual(metrics["false_positives"]['maybe'], 0)
        self.assertEqual(metrics["false_negatives"]['maybe'], 1)
        self.assertAlmostEqual(metrics["accuracy"], 50.0)


        labels = pandas.Series([0, 2, 0, 1, 0, 1])
        predictions = pandas.Series([0, 0, 2, 0, 0, 1])
        metrics = Tools.Metric.compute_metrics_multi_classification(labels, predictions)
        self.assertAlmostEqual(metrics["micro_averaging_accuracy"], 200/3)
        self.assertAlmostEqual(metrics["micro_averaging_precision"], 50.0)
        self.assertAlmostEqual(metrics["micro_averaging_recall"], 50.0)
        self.assertAlmostEqual(metrics["macro_averaging_accuracy"], 200/3)
        self.assertAlmostEqual(metrics["macro_averaging_precision"], 50.0)
        self.assertAlmostEqual(metrics["macro_averaging_recall"], 700/18)
        self.assertEqual(metrics["true_positives"][1], 1)
        self.assertEqual(metrics["true_negatives"][1], 4)
        self.assertEqual(metrics["false_positives"][1], 0)
        self.assertEqual(metrics["false_negatives"][1], 1)
        self.assertAlmostEqual(metrics["accuracy"], 50.0)


    
    def test_prediction_dermatology(self):
        learner = Learning.Scikitlearn("tests/datasets/dermatology.csv", problem_type=Learning.CLASSIFICATION)
        models = learner.evaluate(splitting_method=Learning.K_FOLDS, model_type=Learning.DT, splitting_parameters={"n_models":3, "shuffle":True})
        for id, models in enumerate(models):
            metrics = learner.get_details()[id]["metrics"]
            self.assertTrue("accuracy" in metrics.keys())
            self.assertTrue("precision" in metrics.keys())
            self.assertTrue("recall" in metrics.keys())

    
    def test_prediction_weatherAUS(self):
        learner = Learning.Scikitlearn("tests/datasets/weatherAUS_simplified.csv", problem_type=Learning.CLASSIFICATION)
        models = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.DT, splitting_parameters={'test_size':0.2})
        metrics = learner.get_details()[0]["metrics"]
        self.assertTrue("accuracy" in metrics.keys())
        self.assertTrue("precision" not in metrics.keys())
        self.assertTrue("recall" not in metrics.keys())


    def test_prediction_iris(self):
        learner = Learning.Scikitlearn("tests/datasets/iris.csv", problem_type=Learning.CLASSIFICATION)
        models = learner.evaluate(splitting_method=Learning.LEAVE_ONE_GROUP_OUT, model_type=Learning.RF, splitting_parameters={})
        for id, models in enumerate(models):
            metrics = learner.get_details()[id]["metrics"]
            self.assertTrue("micro_averaging_accuracy" in metrics.keys())
            self.assertTrue("micro_averaging_precision" in metrics.keys())
            self.assertTrue("micro_averaging_recall" in metrics.keys())
            
    def test_import_RF_iris(self):
        self.do_import("tests/datasets/iris.csv", Learning.RF)
        

    def do_import(self, dataset, problem_type):
        data, labels, feature_names = self.load_dataset(dataset)
        results = self.cross_validation(data, labels, problem_type)
        sk_models = [result[0] for result in results]
        training_indexes = [result[1] for result in results]
        test_indexes = [result[2] for result in results]
        
        
        learner, models = Learning.ModelIO.import_models(sk_models, dataset)
        for i, model in enumerate(models):
            instances_details = learner.get_instances(model, indexes=Learning.TEST, test_indexes=test_indexes[i], details=True)
            predictions = [element["prediction"] for element in instances_details]
            true_values = [element["label"] for element in instances_details]
            metrics = learner.compute_metrics(true_values, predictions)
            self.assertTrue("micro_averaging_accuracy" in metrics.keys())
            self.assertTrue("micro_averaging_precision" in metrics.keys())
            self.assertTrue("micro_averaging_recall" in metrics.keys())

    def load_dataset(self, dataset):
        data = pandas.read_csv(dataset).copy()

        # extract labels
        labels = data[data.columns[-1]]
        labels = numpy.array(labels)

        # remove the label of each instance
        data = data.drop(columns=[data.columns[-1]])

        # extract the feature names
        feature_names = list(data.columns)
        return data, labels, feature_names


    def cross_validation(self, X, Y, problem_type, n_trees=5, n_forests=3):
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
            y_train = [Y[x] for x in index_training]
            x_test = X.iloc[index_test]
            y_test = [Y[x] for x in index_test]

            # Training phase
            if problem_type == Learning.RF:
                rf = RandomForestClassifier(n_estimators=n_trees)
            else:
                rf = DecisionTreeClassifier()
            rf.fit(x_train, y_train)

            # Get the classifier prediction of the test set  
            y_predict = rf.predict(x_test)

            forests.append((rf, index_training, index_test))
        return forests
if __name__ == '__main__':
    print("Tests: " + TestMetrics.__name__ + ":")
    unittest.main()
