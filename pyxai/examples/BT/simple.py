from pyxai import Learning, Explainer, Tools

# Check V1.0: Ok

# Machine learning part
learner = Learning.Xgboost(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT)
instance, prediction = learner.get_instances(model=model, n=1, correct=False)

# Explanation part
print(instance)
explainer = Explainer.initialize(model, instance)
direct_reason = explainer.direct_reason()
print("len direct: ", len(direct_reason))
print("is a reason (for 50 checks):", explainer.is_reason(direct_reason, n_samples=50))


tree_specific_reason = explainer.tree_specific_reason(n_iterations=10)
print("\nlen tree_specific: ", len(tree_specific_reason))
print("\ntree_specific: ", explainer.to_features(tree_specific_reason, eliminate_redundant_features=True))
instances = learner.get_instances(n=100)

print(instances)
for inst, p in instances:
    explainer.set_instance(inst)
    direct_reason = explainer.direct_reason()
    
    tree_specific_reason = explainer.tree_specific_reason(n_iterations=100)
    print("is a tree specific", explainer.is_tree_specific_reason(tree_specific_reason))

#explainer.show()
#minimal_tree_specific_reason = explainer.minimal_tree_specific_reason(time_limit=20)
#print("\nlen minimal tree_specific: ", len(minimal_tree_specific_reason))
#print("is a tree specific", explainer.is_tree_specific_reason(minimal_tree_specific_reason))
#if explainer.elapsed_time == Explainer.TIMEOUT: print("Not minimal, this is an approximation")
# s = explainer.sufficient_reason()
