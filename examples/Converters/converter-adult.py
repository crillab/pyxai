# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime
import pandas
data = pandas.read_csv(Tools.Options.dataset, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status', 'occupation', 'relationship','race', 'sex', 'capital-gain','capital-loss', 'hours-per-week', 'native-country', 'salary'])
converter = Learning.Converter(data, target_feature="salary", classification_type=Learning.BINARY_CLASS) # class Converter

print("data:", converter.data)

converter.unset_features(["education-num"])

converter.set_categorical_features(columns=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"])

#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
converter.set_numerical_features({
  "age": None,
  "fnlwgt": None,
  "capital-gain": None,
  "capital-loss": None, 
  "hours-per-week": None
  })

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")
