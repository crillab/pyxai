from pyxai import Learning, Explaining, Tools


Tools.set_verbose(0)

def get_item_iris_short(row):
    row['Width'] = row['Sepal.Width'] + row['Petal.Width']
    row['Length'] = row['Sepal.Length'] + row['Petal.Length']
    row = row.drop(index=['Sepal.Width', 'Petal.Width', 'Sepal.Length', 'Petal.Length'])
    return row
    
def get_item_iris_long(row):
    row['Width'] = row['Sepal.Width'] + row['Petal.Width']
    row['Length'] = row['Sepal.Length'] + row['Petal.Length']
    return row

def get_item_winequelity(row):
    row = row.drop(index=["pH"])
    return row


import unittest
class TestCustomGetItem(unittest.TestCase):
    # TODO : Create a test for each model inside pyxai / Generalize current test
    
    def test_classification_shorter(self):
        learner = Learning.Scikitlearn("tests/datasets/iris.csv", problem_type="classification", instances_type='tabular', get_item_function=get_item_iris_long)
        model = learner.evaluate(splitting_method="hold-out", model_type="decision-tree", splitting_parameters={'test_size':0.2, 'random_state':0}, model_parameters={'random_state':0})
        self.assertEqual(learner.n_features, 6)
        self.assertEqual(model.n_features, 6)
        for x in ['Width', 'Sepal.Width', 'Petal.Width', 'Length', 'Sepal.Length', 'Petal.Length']:
            self.assertIn(x, learner.feature_names)
            self.assertIn(x, model.feature_names)
        instances = learner.get_instances(model=model, n=10, indexes="test")
        for (instance, prediction_raw_model) in instances:
            prediction_pyxai_1 = model.predict_instance(instance)
            implicant = model.instance_to_binaries(instance)
            prediction_pyxai_2 = model.predict_implicant(implicant)
            self.assertEqual(prediction_raw_model, prediction_pyxai_1)
            self.assertEqual(prediction_raw_model, prediction_pyxai_2)


    def test_classification_longer(self):
        learner = Learning.Xgboost("tests/datasets/iris.csv", problem_type="classification", instances_type='tabular', get_item_function=get_item_iris_short)
        models = learner.evaluate(splitting_method="k-folds", model_type="boosted-tree", splitting_parameters={'n_models':3, 'random_state':0}, model_parameters={'random_state':0})
        for model in models:
            self.assertEqual(learner.n_features, 2)
            self.assertEqual(model.n_features, 2)
            for x in ['Width', 'Length']:
                self.assertIn(x, learner.feature_names)
                self.assertIn(x, model.feature_names)
            instances = learner.get_instances(model=model, n=10, indexes="train-in-priority")
            for (instance, prediction_raw_model) in instances:
                prediction_pyxai_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_pyxai_2 = model.predict_implicant(implicant)
                self.assertEqual(prediction_raw_model, prediction_pyxai_1)
                self.assertEqual(prediction_raw_model, prediction_pyxai_2)

    def test_regression_shorter(self):
        learner = Learning.Xgboost("tests/datasets/winequality-red.csv", problem_type="regression", instances_type='tabular', get_item_function=get_item_winequelity)
        models = learner.evaluate(splitting_method="leave-one-group-out", model_type="boosted-tree", splitting_parameters={'n_models':3, 'random_state':0}, model_parameters={'random_state':0})
        for model in models:
            self.assertEqual(learner.n_features, 10)
            self.assertEqual(model.n_features, 10)
            for x in ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']:
                self.assertIn(x, learner.feature_names)
                self.assertIn(x, model.feature_names)
            instances = learner.get_instances(model=model, n=10, indexes="train-in-priority")
            for (instance, prediction_raw_model) in instances:
                prediction_pyxai_1 = model.predict_instance(instance)
                implicant = model.instance_to_binaries(instance)
                prediction_pyxai_2 = model.predict_implicant(implicant)
                self.assertAlmostEqual(prediction_raw_model, prediction_pyxai_1, 5)
                self.assertAlmostEqual(prediction_raw_model, prediction_pyxai_2, 5)


        
if __name__ == '__main__':
    print("Tests: " + TestCustomGetItem.__name__ + ":")
    unittest.main()