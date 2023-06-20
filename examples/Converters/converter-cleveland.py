# https://www.openml.org/search?type=data&status=active&id=786&sort=runs
from pyxai import Learning, Explainer, Tools

import datetime

preprocessor = Learning.Preprocessor(Tools.Options.dataset, target_feature="binaryClass", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

preprocessor.set_categorical_features(columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"])

#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
preprocessor.set_numerical_features({
  "age": None,
  "trestbps": None,
  "chol": None,
  "thalach": None,
  "oldpeak": None,
  "ca": None,
  })

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")

