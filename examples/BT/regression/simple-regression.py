#python3 examples/BT/regression/simple-regression.py -dataset=tests/winequality-red.csv 
## Check V1.0: Ok

from pyxai import Learning, Explainer, Tools

# Machine learning part
learner = Learning.Xgboost(Tools.Options.dataset,  learner_type=Learning.REGRESSION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT)
instances= learner.get_instances(model=model, n=10)
i = 0
instance = instances[i][0]
p = instances[i][1]



# Explanation part
print("instance", instance)

explainer = Explainer.initialize(model, instance)
prediction = explainer.predict(instance)
print("prediction: ", explainer.predict(instance))
print("learner prediction: ", p)
extremum_range = explainer.extremum_range()
Lf = extremum_range[1] - extremum_range[0]
print("extremum range: ",extremum_range)

direct_reason = explainer.direct_reason()
print("direct: ", explainer.to_features(direct_reason))
print("len direct: ", len(direct_reason), len(explainer.to_features(direct_reason)))
#print("is a reason (for 50 checks):", explainer.is_reason(direct_reason))


percent = 2.5

delta = (percent/100)*Lf
print("delta=", delta)
explainer.set_interval(prediction - delta, prediction + delta)
print(f"set interval to : [", explainer.lower_bound, explainer.upper_bound, "]")
tree_specific_reason = explainer.tree_specific_reason(n_iterations=1)
print("\nlen tree_specific: ", len(tree_specific_reason), len(explainer.to_features(tree_specific_reason, eliminate_redundant_features=True)))
print("\ntree_specific: ", explainer.to_features(tree_specific_reason, eliminate_redundant_features=True))
print("is a tree specific", explainer.is_tree_specific_reason(tree_specific_reason))




print("\n\nActivate theorie")
explainer = Explainer.initialize(model, instance=instance, features_type={"numerical": Learning.DEFAULT})
direct_reason = explainer.direct_reason()
print("len direct: ", len(direct_reason))
#print("is a reason (for 50 checks):", explainer.is_reason(direct_reason))
explainer.set_interval(prediction - delta, prediction + delta)
print(f"set interval to [{1-percent} * prediction, {1+percent} * prediction]: [", explainer.lower_bound, explainer.upper_bound, "]")
#
tree_specific_reason = explainer.tree_specific_reason(n_iterations=1000)
print("\ntree_specific: ", tree_specific_reason)
print("\nlen tree_specific: ", len(tree_specific_reason))
print("\ntree_specific: ", explainer.to_features(tree_specific_reason, eliminate_redundant_features=True))
print("is a tree specific", explainer.is_tree_specific_reason(tree_specific_reason))

explainer.show()