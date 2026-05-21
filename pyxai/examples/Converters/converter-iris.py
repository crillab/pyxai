# dataset source: http://archive.ics.uci.edu/dataset/53/iris


import pandas
from pyxai import Learning, Tools

data = pandas.read_csv(Tools.Options.dataset, names=["sepal length", "sepal width", "petal length", "petal width", "Iris Plants"])
preprocessor = Learning.TabularPreprocessor(data, target_feature="Iris Plants", problem_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

preprocessor.all_numerical_features()
preprocessor.process()

dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory=Tools.Options.output)
