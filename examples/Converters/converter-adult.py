# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime
import pandas
data = pandas.read_csv(Tools.Options.dataset, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status', 'occupation', 'relationship','race', 'sex', 'capital-gain','capital-loss', 'hours-per-week', 'native-country', 'salary'])
preprocessor = Learning.Preprocessor(data, target_feature="salary", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

print("data:", preprocessor.data)

preprocessor.unset_features(["education-num"])

preprocessor.set_categorical_features(columns=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"])

preprocessor.set_numerical_features({"age": None, "fnlwgt": None, "capital-gain": None, "capital-loss": None, "hours-per-week": None})

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")
