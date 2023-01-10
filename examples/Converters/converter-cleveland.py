# https://www.openml.org/search?type=data&status=active&id=786&sort=runs
from pyxai import Learning, Explainer, Tools

import datetime

converter = Learning.Converter(Tools.Options.dataset, target_feature="binaryClass", classification_type=Learning.BINARY_CLASS) # class Converter

converter.set_categorical_features(columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"])

#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
converter.set_numerical_features({
  "age": None,
  "trestbps": None,
  "chol": None,
  "thalach": None,
  "oldpeak": None,
  "ca": None,
  })

#print(converter.get_types())
converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")

