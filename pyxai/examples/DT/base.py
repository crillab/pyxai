# Check V2.0: Not Ok with 100% of accuracy it was unable to select an incorrect instance...

from pyxai import Learning, Explaining

learner = Learning.Scikitlearn("./tests/dermatology.csv", problem_type='classification', instances_type='tabular')
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type='decision-tree')
instance, prediction = learner.get_instances(model, n=1, is_correct=True, subset_predicted_classes=[0])

explainer = Explaining.initialize(model, instance)
print("instance:", instance)
print("binary representation:", explainer.binary_representation)

sufficient_reason = explainer.sufficient_reason(n=1)
print("sufficient_reason:", sufficient_reason)
print("to_features:", explainer.to_features(sufficient_reason))

instance, prediction = learner.get_instances(model, n=1, is_correct=False)
explainer.set_instance(instance)
contrastive_reason = explainer.contrastive_reason()
print("contrastive reason", contrastive_reason)
print("to_features:", explainer.to_features(contrastive_reason, contrastive=True))

explainer.visualisation.screen(instance, contrastive_reason, contrastive=True)