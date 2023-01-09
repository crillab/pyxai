# dataset source: https://archive.ics.uci.edu/ml/datasets/spambase

from pyxai import Learning, Explainer, Tools

import datetime


import pandas
data = pandas.read_csv(Tools.Options.dataset, names=["V"+i for i in range(58)])

converter = Learning.Converter(data, target_feature="V57", classification_type=Learning.BINARY_CLASS) # class Converter

converter.all_numerical_features()

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")
