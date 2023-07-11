# dataset source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

from pyxai import Learning, Explainer, Tools

import datetime
import pandas

data = pandas.read_csv(Tools.Options.dataset, index_col=False) #Warning, here there are no index columns

print("data:", data)
preprocessor = Learning.Preprocessor(data, target_feature="V20", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)
preprocessor.set_categorical_features(columns=["V1", "V3", "V4", "V6", "V7", "V9", "V10","V12", "V14", "V15", "V17", "V19"])

preprocessor.set_numerical_features({
  "V2": None,
  "V5": None,
  "V8": None,
  "V11": None,
  "V13": None,
  "V16": None,
  "V18": None,
})

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")
