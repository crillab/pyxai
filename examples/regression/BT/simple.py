from pyxai import Learning, Explainer, Tools

# Machine learning part
learner = Learning.Xgboost(Tools.Options.dataset)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT, learner_type=Learning.REGRESSION)
instance, prediction = learner.get_instances(model=model, n=1, correct=False)

# Explanation part
print("instance", instance)
print("prediction", prediction)

explainer = Explainer.initialize(model, instance)
print("extremum range: ",explainer.extremum_range())

direct_reason = explainer.direct_reason()
print("len direct: ", len(direct_reason))
print("is a reason (for 50 checks):", explainer.is_reason(direct_reason))


percent = 0.2

explainer.set_range(prediction*(1-percent), prediction*(1+percent))
print(f"set interval to [{1-percent} * prediction, {1+percent} * prediction]: [", explainer.lower_bound, explainer.upper_bound, "]")

tree_specific_reason = explainer.tree_specific_reason(n_iterations=1)
print("\ntree_specific: ", tree_specific_reason)
print("\nlen tree_specific: ", len(tree_specific_reason))
print("\ntree_specific: ", explainer.to_features(tree_specific_reason, eliminate_redundant_features=True))
print("is a tree specific", explainer.is_tree_specific_reason(tree_specific_reason))

print("\n\nActivate theorie")
explainer = Explainer.initialize(model, instance=instance, features_types={"numerical": Learning.DEFAULT})
direct_reason = explainer.direct_reason()
print("len direct: ", len(direct_reason))
print("is a reason (for 50 checks):", explainer.is_reason(direct_reason))

explainer.set_range(prediction*(1-percent), prediction*(1+percent))
print(f"set interval to [{1-percent} * prediction, {1+percent} * prediction]: [", explainer.lower_bound, explainer.upper_bound, "]")

tree_specific_reason = explainer.tree_specific_reason(n_iterations=1)
print("\ntree_specific: ", tree_specific_reason)
print("\nlen tree_specific: ", len(tree_specific_reason))
print("\ntree_specific: ", explainer.to_features(tree_specific_reason, eliminate_redundant_features=True))
print("is a tree specific", explainer.is_tree_specific_reason(tree_specific_reason))
