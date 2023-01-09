# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime

converter = Learning.Converter(Tools.Options.dataset, target_feature="A15", classification_type=Learning.BINARY_CLASS) # class Converter

print("data:", converter.data)


converter.set_categorical_features(columns=["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"])

#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
converter.set_numerical_features({
  "A2": None,
  "A3": None,
  "A7": None,
  "A10": None,
  "A13": None,
  "A14": None,
  })

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0]
converter.export(dataset_name, output="examples/datasets_converted")
