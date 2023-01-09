# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime
import pandas
data = pandas.read_csv(Tools.Options.dataset, names=['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt','drinks', 'selector'])
converter = Learning.Converter(data, target_feature="drinks", classification_type=Learning.BINARY_CLASS) # class Converter

print("data:", converter.data)

converter.unset_features(["selector"])

converter.all_numerical_features()

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")
