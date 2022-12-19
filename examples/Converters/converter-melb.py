# usage
# python3 examples/Converters/converter-melb.py -dataset=../../melb_data.csv

from pyxai import Learning, Explainer, Tools

import datetime

# Machine learning part
# NUMERICAL: with an order (Ordinal Encoding) 
# CATEGORICAL: without an order 

converter = Learning.Converter(Tools.Options.dataset, target_feature="Type", classification_type=Learning.BINARY_CLASS) # class Converter

print(converter.data)
exit(0)
converter.set_categorical_features(columns_name=["Suburb", "Address", "Type", "Method", "SellerG", "Postcode", "CouncilArea", "Regionname"])

#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
converter.set_numerical_features({
  "Rooms":None, 
  "Price":None,
  "Date": lambda d: datetime.date(int(d.split("/")[2]), int(d.split("/")[1]), int(d.split("/")[0])).toordinal(), 
  "Distance": None,
  "Bedroom2": None,
  "Bathroom": None,
  "Car": None,
  "Landsize": None,
  "BuildingArea": None,
  "YearBuilt": None,
  "Lattitude": None,
  "Longtitude": None,
  "Propertycount": None
  })

#print(converter.get_types())
converter.process()

dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name+".csv")


learner = Learning.Scikitlearn(dataset_name+".csv", types=dataset_name+".types")
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
instance, prediction = learner.get_instances(model=model, n=1, correct=False)


explainer = Explainer.initialize(model, instance=instance)
