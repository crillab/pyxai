# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime


import pandas

preprocessor = Learning.Preprocessor(Tools.Options.dataset, target_feature="784", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS) 
print("data:", preprocessor.data)

preprocessor.all_numerical_features()

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory=Tools.Options.output)
