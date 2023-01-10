# dataset source: https://archive.ics.uci.edu/ml/datasets/cnae-9

from pyxai import Learning, Explainer, Tools

import datetime
import pandas
data = pandas.read_csv(Tools.Options.dataset, names=['category']+["W"+str(i) for i in range(856)])
converter = Learning.Converter(data, target_feature="category", classification_type=Learning.BINARY_CLASS) # class Converter

print("data:", converter.data)

converter.all_numerical_features()

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")
