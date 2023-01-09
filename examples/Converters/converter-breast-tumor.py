# dataset source: https://www.openml.org/search?type=data&status=active&id=844&sort=runs

from pyxai import Learning, Explainer, Tools

import datetime

converter = Learning.Converter(Tools.Options.dataset, target_feature="binaryClass", classification_type=Learning.BINARY_CLASS) # class Converter

print("data:", converter.data)


converter.set_categorical_features(columns=["menopause", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad","irradiation", "recurrence"])

#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
converter.set_numerical_features({ "age": None })

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0]
converter.export(dataset_name, output="examples/datasets_converted")
