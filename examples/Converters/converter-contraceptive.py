# dataset source: https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice

from pyxai import Learning, Explainer, Tools

import datetime
import pandas
data = pandas.read_csv(Tools.Options.dataset, names=['age', 'wife-education', 'husband-education', 'children', 'religion','job', 'occupation', 'index','media', 'contraceptive_method'])

converter = Learning.Converter(data, target_feature="contraceptive_method", classification_type=Learning.BINARY_CLASS) # class Converter

print("data:", converter.data)


converter.set_categorical_features(columns=["wife-education","husband-education","religion","job","occupation","index","media"])

#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
converter.set_numerical_features({ "age": None, "children": None })

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0]
converter.export(dataset_name, output="examples/datasets_converted")
