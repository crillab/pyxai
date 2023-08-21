from pyxai import Learning, Explainer, Tools

# usage
# python3 examples/RF/theories-types-file.py -dataset=examples/datasets_converted/australian_0.csv -types=examples/datasets_converted/australian_0.types
# Check V1.0: Ok 

# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
instances = learner.get_instances(model, n=2)

#print("instance:", instance)


# Explainer part
australian_types = {
    "numerical": Learning.DEFAULT,
    "categorical": {"A4*": (1, 2, 3), 
                    "A5*": tuple(range(1, 15)),
                    "A6*": (1, 2, 3, 4, 5, 7, 8, 9), 
                    "A12*": tuple(range(1, 4))},
    "binary": ["A1", "A8", "A9", "A11"],
}

explainer = Explainer.initialize(model, features_type=australian_types)
for (instance, prediction) in instances:
    explainer.set_instance(instance)

    majoritary_reason = explainer.majoritary_reason(n_iterations=10)
    print("\nlen tree_specific: ", len(majoritary_reason))
    print("\ntree_specific: ", explainer.to_features(majoritary_reason, eliminate_redundant_features=True))
    print("is a tree specific", explainer.is_majoritary_reason(majoritary_reason))

    contrastive = explainer.minimal_contrastive_reason(time_limit=100)
    features = explainer.to_features(contrastive, contrastive=True)

    print("contrastive:", contrastive)
    print("features contrastive:", features)

explainer.show()