# dataset source: https://www.openml.org/search?type=data&sort=runs&id=4136&status=active

from pyxai import Learning, Explainer, Tools

converter = Learning.Converter(Tools.Options.dataset, target_feature="20000", classification_type=Learning.BINARY_CLASS) # class Converter

converter.all_numerical_features()

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")


