# dataset source: https://www.openml.org/search?type=data&status=active&id=41142&sort=runs

from pyxai import Learning, Explainer, Tools

preprocessor = Learning.Preprocessor(Tools.Options.dataset, target_feature="1636", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

preprocessor.all_numerical_features()

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")


