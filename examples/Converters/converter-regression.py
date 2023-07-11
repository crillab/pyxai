# dataset source: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

from pyxai import Learning, Explainer, Tools

import datetime
import pandas
data = pandas.read_csv(Tools.Options.dataset)
preprocessor = Learning.Preprocessor(data, target_feature="quality", learner_type=Learning.REGRESSION)

print("data:", preprocessor.data)

preprocessor.all_numerical_features()

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")
