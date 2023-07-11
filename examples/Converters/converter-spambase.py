# dataset source: https://archive.ics.uci.edu/ml/datasets/spambase

from pyxai import Learning, Explainer, Tools

import datetime


import pandas
data = pandas.read_csv(Tools.Options.dataset, names=["V"+str(i) for i in range(58)])

preprocessor = Learning.Preprocessor(data, target_feature="V57", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

preprocessor.all_numerical_features()

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")
