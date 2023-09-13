# dataset source: https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice

from pyxai import Learning, Explainer, Tools

import datetime
import pandas
data = pandas.read_csv(Tools.Options.dataset, names=['age', 'wife-education', 'husband-education', 'children', 'religion','job', 'occupation', 'index','media', 'contraceptive_method'])

preprocessor = Learning.Preprocessor(data, target_feature="contraceptive_method", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

print("data:", preprocessor.data)


preprocessor.set_categorical_features(columns=["wife-education","husband-education","religion","job","occupation","index","media"])

preprocessor.set_numerical_features({ "age": None, "children": None })

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0]
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")
