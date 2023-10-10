# dataset source: https://www.kaggle.com/datasets/danofer/compass

from pyxai import Learning, Explainer, Tools

preprocessor = Learning.Preprocessor(Tools.Options.dataset, target_feature="Two_yr_Recidivism", learner_type=Learning.CLASSIFICATION, classification_type=Learning.BINARY_CLASS)


print("data:", preprocessor.data)

preprocessor.set_categorical_features_already_one_hot_encoded("score_factor", ["score_factor"])
preprocessor.set_categorical_features_already_one_hot_encoded("Age_Above_FourtyFive", ["Age_Above_FourtyFive"])
preprocessor.set_categorical_features_already_one_hot_encoded("Age_Below_TwentyFive", ["Age_Below_TwentyFive"])
preprocessor.set_categorical_features_already_one_hot_encoded("Ethnic", ["African_American", "Asian", "Hispanic", "Native_American", "Other"])
preprocessor.set_categorical_features_already_one_hot_encoded("Female", ["Female"])
preprocessor.set_categorical_features_already_one_hot_encoded("Misdemeanor", ["Misdemeanor"])

preprocessor.set_numerical_features({
  "Number_of_Priors": None
})

preprocessor.process()
dataset_name = Tools.Options.dataset.split("/")[-1].split(".")[0] 
preprocessor.export(dataset_name, output_directory="examples/datasets_converted")
