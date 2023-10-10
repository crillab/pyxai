# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime

preprocessor = Learning.Preprocessor(Tools.Options.dataset, target_feature="A15", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

print("data:", preprocessor.data)


preprocessor.set_categorical_features(columns=["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"])

#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
preprocessor.set_numerical_features({
  "A2": None,
  "A3": None,
  "A7": None,
  "A10": None,
  "A13": None,
  "A14": None,
  })

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0]
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")
