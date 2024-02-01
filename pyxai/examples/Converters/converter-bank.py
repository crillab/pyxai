# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import pandas
data = pandas.read_csv(Tools.Options.dataset, names=["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"], skiprows=1, sep=";") 

preprocessor = Learning.Preprocessor(data, target_feature="y", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

print("data:", preprocessor.data)

preprocessor.unset_features([])

preprocessor.set_categorical_features(columns=["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"])

preprocessor.set_numerical_features({
  "age": None,
  "balance": None,
  "day": None, 
  "duration": None, 
  "campaign": None, 
  "pdays": None,
  "previous": None
  })

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory=Tools.Options.output)
