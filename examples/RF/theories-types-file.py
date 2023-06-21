from pyxai import Learning, Explainer, Tools

# usage
# python3 examples/RF/theories-types-file.py -dataset=examples/datasets_converted/australian_0.csv -types=examples/datasets_converted/australian_0.types


# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
instance, prediction = learner.get_instances(n=1)

print("instance:", instance)


# Explainer part
explainer = Explainer.initialize(model, instance=instance, features_type=Tools.Options.types)


contrastive = explainer.minimal_contrastive_reason(time_limit=100)
features = explainer.to_features(contrastive, contrastive=True)

print("contrastive:", contrastive)
print("features contrastive:", features)

majoritary_reason = explainer.majoritary_reason(n_iterations=10)
features = explainer.to_features(majoritary_reason)
print("features majoritary:", features)

