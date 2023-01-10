# dataset source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

from pyxai import Learning, Explainer, Tools

import datetime
import pandas

data = pandas.read_csv(Tools.Options.dataset, index_col=False) #Warning, here there are no index columns

print("data:", data)
converter = Learning.Converter(data, target_feature="V20", classification_type=Learning.BINARY_CLASS) # class Converter
converter.set_categorical_features(columns=["V1", "V3", "V4", "V6", "V7", "V9", "V10","V12", "V14", "V15", "V17", "V19"])

converter.set_numerical_features({
  "V2": None,
  "V5": None,
  "V8": None,
  "V11": None,
  "V13": None,
  "V16": None,
  "V18": None,
})

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")
