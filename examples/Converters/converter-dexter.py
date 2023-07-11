# dataset source: https://www.openml.org/search?type=data&sort=runs&id=4136&status=active

from pyxai import Learning, Explainer, Tools

preprocessor = Learning.Preprocessor(Tools.Options.dataset, target_feature="20000", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

preprocessor.all_numerical_features()

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")


