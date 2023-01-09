# dataset source: https://www.kaggle.com/datasets/danofer/compass

from pyxai import Learning, Explainer, Tools

converter = Learning.Converter(Tools.Options.dataset, target_feature="Two_yr_Recidivism", classification_type=Learning.BINARY_CLASS) # class Converter


print("data:", converter.data)

converter.set_categorical_features_already_one_hot_encoded("score_factor", ["score_factor"])
converter.set_categorical_features_already_one_hot_encoded("Age_Above_FourtyFive", ["Age_Above_FourtyFive"])
converter.set_categorical_features_already_one_hot_encoded("Age_Below_TwentyFive", ["Age_Below_TwentyFive"])
converter.set_categorical_features_already_one_hot_encoded("Ethnic", ["African_American", "Asian", "Hispanic", "Native_American", "Other"])
converter.set_categorical_features_already_one_hot_encoded("Female", ["Female"])
converter.set_categorical_features_already_one_hot_encoded("Misdemeanor", ["Misdemeanor"])



#datetime.date(d.split("/")[2], d.split("/")[1], d.split("/")[0]).toordinal()
converter.set_numerical_features({
  "Number_of_Priors": None
})

converter.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
converter.export(dataset_name, output="examples/datasets_converted")
