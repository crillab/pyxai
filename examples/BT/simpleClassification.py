from pyxai import Learning, Explainer, Tools

# Machine learning part
learner = Learning.Xgboost(Tools.Options.dataset)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT)
instance, prediction = learner.get_instances(model=model, n=1, correct=False)


# Explanation part
print(instance)
explainer = Explainer.initialize(model, instance)
direct_reason = explainer.direct_reason()
print("len direct: ", len(direct_reason))
print("is a reason (for 50 checks):", explainer.is_reason(direct_reason, n_samples=50))


tree_specific_reason = explainer.tree_specific_reason(n_iterations=10)

print("tree_specific_reason:", tree_specific_reason)
print("tree_specific_reason:", explainer.to_features(tree_specific_reason))

print("is a reason (for 50 checks):", explainer.is_tree_specific_reason(tree_specific_reason))
