# dataset source: https://archive.ics.uci.edu/ml/datasets/adult

from pyxai import Learning, Explainer, Tools

import datetime

converter = Learning.Converter(Tools.Options.dataset, target_feature="default.payment.next.month", classification_type=Learning.BINARY_CLASS) # class Converter

print("data:", converter.data)

converter.unset_features(["id"])

#,LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,default.payment.next.month
converter.set_categorical_features(columns=["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4","PAY_5", "PAY_6"])

converter.set_numerical_features({
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
  "PAY_AMT14": None,
  "PAY_AMT5": None,
  "PAY_AMT6": None
})

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")
