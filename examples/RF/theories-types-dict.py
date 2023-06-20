from pyxai import Learning, Explainer, Tools

# usage
# python3 examples/RF/theories-types-file.py -dataset=examples/datasets_converted/australian_0.csv -types=examples/datasets_converted/australian_0.types


# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
instance, prediction = learner.get_instances(model, n=1, seed=11200, correct=False)

#print("instance:", instance)


# Explainer part
australian_types = {
    "numerical": Learning.DEFAULT,
    "categorical": ["A4*", "A5*", "A6*", "A12*"],
    "binary": ["A1", "A8", "A9", "A11"],
}
explainer = Explainer.initialize(model, instance=instance)
print("No theory")
majoritary_reason = explainer.majoritary_reason(n_iterations=10)
print("\nlen tree_specific: ", len(majoritary_reason))
print("\ntree_specific: ", explainer.to_features(majoritary_reason, eliminate_redundant_features=True))
print("is a tree specific", explainer.is_majoritary_reason(majoritary_reason))

print("instance: ", instance)
print("Theory")
explainer = Explainer.initialize(model, instance=instance, features_type=australian_types)
print("OK")
majoritary_reason = explainer.majoritary_reason(n_iterations=10)
print("\nlen tree_specific: ", len(majoritary_reason))
print("\ntree_specific: ", explainer.to_features(majoritary_reason, eliminate_redundant_features=True))
print("is a tree specific", explainer.is_majoritary_reason(majoritary_reason))

contrastive = explainer.minimal_contrastive_reason(time_limit=100)
features = explainer.to_features(contrastive, contrastive=True)

print("contrastive:", contrastive)
print("features contrastive:", features)
