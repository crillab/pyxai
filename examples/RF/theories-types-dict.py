from pyxai import Learning, Explainer, Tools

# usage
# python3 examples/RF/theories-types-file.py -dataset=examples/datasets_converted/australian_0.csv -types=examples/datasets_converted/australian_0.types


# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
instance, prediction = learner.get_instances(n=1)

#print("instance:", instance)


# Explainer part
australian_types = {
    "numerical": Learning.DEFAULT,
    "categorical": ["A4*", "A5*", "A6*", "A12*"],
    "binary": ["A1", "A8", "A9", "A11"],
}

explainer = Explainer.initialize(model, instance=instance, features_types=australian_types)


contrastive = explainer.minimal_contrastive_reason(time_limit=100)
features = explainer.to_features(contrastive, eliminate_redundant_features=True, inverse=True)

print("contrastive:", contrastive)
print("features contrastive:", features)
