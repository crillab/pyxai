from pyxai import Learning, Explainer, Tools

# usage
#python3 examples/RF/theories-majoritary.py -dataset=../../data/dataMLothers/iris.csv 

# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
instance, prediction = learner.get_instances(n=1)

print("instance:", instance)

# Explainer part
print("No theory")
explainer = Explainer.initialize(model, instance=instance)
majoritary_reason = explainer.majoritary_reason(n_iterations=10)
print("majoritary:", majoritary_reason)

print("\nlen majoritary: ", len(majoritary_reason))
print("\nmajoritary: ", explainer.to_features(majoritary_reason, eliminate_redundant_features=False))
print("is a majoritary", explainer.is_majoritary_reason(majoritary_reason))

print("instance: ", instance)
print("Theory")


explainer = Explainer.initialize(model, instance=instance, features_types={"numerical": Learning.DEFAULT})

direct_reason = explainer.direct_reason()
print("len direct:", len(direct_reason))
#print("is a reason (for 50 checks):", explainer.is_reason(direct_reason, n_samples=50))

majoritary = explainer.majoritary_reason(n=1, n_iterations=1)
print("majoritary:", majoritary)

print("\nmajoritary: ", explainer.to_features(majoritary, eliminate_redundant_features=False))
print("\nmajoritary: ", explainer.to_features(majoritary))

print("\nlen majoritary: ", len(majoritary))

print("is_majoritary_reason (for 50 checks):", explainer.is_majoritary_reason(majoritary))

exit(0)
# can be costly
sufficient_reason = explainer.sufficient_reason(time_limit=5)
print("\nlen sufficient reason:", len(sufficient_reason))
if explainer.elapsed_time == Explainer.TIMEOUT: print("Time out, this is an approximation")
print("is reason (for 50 checks)", explainer.is_reason(sufficient_reason, n_samples=50))

minimal_constrative_reason = explainer.minimal_contrastive_reason(time_limit=5)
if len(minimal_constrative_reason) == 0:
    print("\nminimal contrastive not found")
else:
    print("\nminimal contrastive: ", len(minimal_constrative_reason))
    if explainer.elapsed_time == Explainer.TIMEOUT: print("Time out, this is an approximation")
    print("is  contrastive: ", explainer.is_contrastive_reason(minimal_constrative_reason))
