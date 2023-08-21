# dataset source: https://www.openml.org/search?type=data&status=active&id=844&sort=runs

from pyxai import Learning, Explainer, Tools

import datetime

preprocessor = Learning.Preprocessor(Tools.Options.dataset, target_feature="binaryClass", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

print("data:", preprocessor.data)


preprocessor.set_categorical_features(columns=["menopause", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad","irradiation", "recurrence"])

#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
preprocessor.set_numerical_features({ "age": None })

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0]
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")
