# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime

preprocessor = Learning.Preprocessor(Tools.Options.dataset, target_feature="default.payment.next.month", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)

preprocessor.unset_features(["ID"])

#,LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,default.payment.next.month
preprocessor.set_categorical_features(columns=["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4","PAY_5", "PAY_6"])

preprocessor.set_numerical_features({
  "LIMIT_BAL": None,
  "AGE": None,
  "BILL_AMT1": None,
  "BILL_AMT2": None,
  "BILL_AMT3": None,
  "BILL_AMT4": None,
  "BILL_AMT5": None,
  "BILL_AMT6": None,
  "PAY_AMT1": None,
  "PAY_AMT2": None,
  "PAY_AMT3": None,
  "PAY_AMT4": None,
  "PAY_AMT5": None,
  "PAY_AMT6": None
})

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")
