# dataset source:http://www.timeseriesclassification.com/description.php?Dataset=ArrowHead

from pyxai import Learning, Explainer, Tools

import pandas

data = pandas.read_csv(Tools.Options.dataset, names=["V"+str(i) for i in range(1,250)]+["label"], sep=',') 

preprocessor = Learning.Preprocessor(data, target_feature="label", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)


preprocessor.all_numerical_features()

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory=Tools.Options.output)
