from pyxai import Learning, Explainer, Tools
import math

# Regression
#learner = Learning.Xgboost("tests/winequality-red.csv")
learner = Learning.Xgboost("tests/iris.csv")

models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2, learner_type=Learning.REGRESSION, base_score=0, n_estimators=5)

#for model in [models[0]]:
for model in models:
    instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
    for i, (instance, prediction_classifier) in enumerate(instances):
        
        #if i == 2:
            prediction_classifier = round(prediction_classifier,4)
            prediction_model_1 = round(model.predict_instance(instance),4)
            implicant = model.instance_to_binaries(instance)
            prediction_model_2 = round(model.predict_implicant(implicant),4)
            
            if (str(prediction_classifier) != str(prediction_model_1)) or (str(prediction_classifier) != str(prediction_model_2)):
                print("instance:", instance)
                print("prediction_classifier:", prediction_classifier)
                print("prediction_model_1:", prediction_model_1)
                print("prediction_model_2:", prediction_model_2)
                exit(0)

        
        #assert str(prediction_classifier) == str(prediction_model_1) and 
print("OK")
exit(0)
# Binary classes
learner = Learning.Xgboost("tests/dermatology.csv")
models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)

for model in models:
    instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
    for (instance, prediction_classifier) in instances:
        prediction_model_1 = model.predict_instance(instance)
        implicant = model.instance_to_binaries(instance)
        prediction_model_2 = model.predict_implicant(implicant)
        assert prediction_classifier == prediction_model_1 and prediction_classifier == prediction_model_2

# Multi-classes
learner = Learning.Xgboost("tests/iris.csv")
models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, test_size=0.2)

for model in models:
    instances = learner.get_instances(model=model, n=10, indexes=Learning.TEST)
    for (instance, prediction_classifier) in instances:
        prediction_model_1 = model.predict_instance(instance)
        implicant = model.instance_to_binaries(instance)
        prediction_model_2 = model.predict_implicant(implicant)
        assert prediction_classifier == prediction_model_1 and prediction_classifier == prediction_model_2

        
