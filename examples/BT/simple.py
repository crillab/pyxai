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

tree_specific_reason = explainer.tree_specific_reason()
print("\nlen tree_specific: ", len(tree_specific_reason))
print("is a tree specific", explainer.is_tree_specific_reason(tree_specific_reason))

minimal_tree_specific_reason = explainer.minimal_tree_specific_reason(time_limit=20)
print("\nlen minimal tree_specific: ", len(minimal_tree_specific_reason))
print("is a tree specific", explainer.is_tree_specific_reason(minimal_tree_specific_reason))
if explainer.elapsed_time == Explainer.TIMEOUT: print("Not minimal, this is an approximation")
# s = explainer.sufficient_reason()
