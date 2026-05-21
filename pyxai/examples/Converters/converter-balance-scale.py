# dataset source: https://archive.ics.uci.edu/ml/datasets/balance+scale

from pyxai import Learning, Explainer, Tools

import datetime
import pandas
data = pandas.read_csv(Tools.Options.dataset, names=['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'])
preprocessor = Learning.TabularPreprocessor(data, target_feature="Class", problem_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

print("data:", preprocessor.data)

preprocessor.all_numerical_features()

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory=Tools.Options.output)
