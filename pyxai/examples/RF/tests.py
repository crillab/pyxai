from pyxai import Learning, Explaining, Tools


learner = Learning.Scikitlearn("tests/datasets/iris.csv", problem_type='classification')
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.DT, splitting_parameters={'random_state':0})
instance, prediction = learner.get_instances(model, n=1, is_correct=True, subset_predicted_classes=["Iris-setosa"], seed=2)
explainer = Explaining.initialize(model, instance)
print("instance:", instance)
print("binary representation:", explainer.binary_representation)
print("conditions related to the binary representation:", explainer.to_features(explainer.binary_representation,eliminate_redundant_features=False))
sufficient_reason = explainer.sufficient_reason(n=1)
print("sufficient_reason:", sufficient_reason)
print("to_features:", explainer.to_features(sufficient_reason))
print("class:", type(explainer))
print("is sufficient: ", explainer.is_sufficient_reason(sufficient_reason))