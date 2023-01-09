# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime


import pandas

converter = Learning.Converter(Tools.Options.dataset, target_feature="5000", classification_type=Learning.BINARY_CLASS) # class Converter

converter.all_numerical_features()

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")
