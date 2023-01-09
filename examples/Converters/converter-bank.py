# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime

converter = Learning.Converter(Tools.Options.dataset, target_feature="y", classification_type=Learning.BINARY_CLASS) # class Converter

print("data:", converter.data)

converter.unset_features([])

converter.set_categorical_features(columns=["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"])

#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
converter.set_numerical_features({
  "age": None,
  "balance": None,
  "day": None, 
  "duration": None, 
  "campaign": None, 
  "pdays": None,
  "previous": None
  })

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")
