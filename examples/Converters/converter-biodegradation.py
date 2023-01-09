# dataset source: https://www.openml.org/search?type=data&status=active&id=1494&sort=runs

from pyxai import Learning, Explainer, Tools

import datetime


import pandas

converter = Learning.Converter(Tools.Options.dataset, target_feature="41", classification_type=Learning.BINARY_CLASS) 

converter.all_numerical_features()

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")
