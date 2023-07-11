from pyxai import Learning, Explainer, Tools

# usage
# python3 pyxai/examples/RF/Simple.py -dataset=path/to/dataset.csv
# Check V1.0: Ok 

# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
instance, prediction = learner.get_instances(n=1)

print("instance:", instance)

# Explainer part
explainer = Explainer.initialize(model, instance=instance, features_type=Tools.Options.types)

direct_reason = explainer.direct_reason()
print("len direct:", len(direct_reason))
print("is a reason (for 50 checks):", explainer.is_reason(direct_reason, n_samples=50))

majoritary = explainer.majoritary_reason(n=1, n_iterations=1)
print("\nmajoritary: ", explainer.to_features(majoritary))
print("\nlen majoritary: ", len(majoritary))
print("is_majoritary_reason (for 50 checks):", explainer.is_majoritary_reason(majoritary))

minimal_majoritary = explainer.minimal_majoritary_reason(time_limit=10)

print("\nminimal majoritary: ", explainer.to_features(minimal_majoritary))
print("len majoritary: ", len(minimal_majoritary))
print("is_majoritary_reason (for 50 checks):", explainer.is_majoritary_reason(minimal_majoritary))


# can be costly
sufficient_reason = explainer.sufficient_reason(time_limit=5)
print("\nlen sufficient reason:", len(sufficient_reason))
if explainer.elapsed_time == Explainer.TIMEOUT: print("Time out, this is an approximation")
print("sufficient: ", explainer.to_features(sufficient_reason))
print("is reason (for 50 checks)", explainer.is_reason(sufficient_reason, n_samples=50))

minimal_constrative_reason = explainer.minimal_contrastive_reason(time_limit=5)
if len(minimal_constrative_reason) == 0:
    print("\nminimal contrastive not found")
else:
    print("\nminimal contrastive: ", len(minimal_constrative_reason))
    if explainer.elapsed_time == Explainer.TIMEOUT: print("Time out, this is an approximation")
    print("is  contrastive: ", explainer.is_contrastive_reason(minimal_constrative_reason))

explainer.show()