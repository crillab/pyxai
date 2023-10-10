# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime

preprocessor = Learning.Preprocessor(Tools.Options.dataset, target_feature="y", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

print("data:", preprocessor.data)

preprocessor.unset_features([])

preprocessor.set_categorical_features(columns=["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"])

preprocessor.set_numerical_features({
  "age": None,
  "balance": None,
  "day": None, 
  "duration": None, 
  "campaign": None, 
  "pdays": None,
  "previous": None
  })

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")
