from pyxai import Learning, Explaining, Tools

# usage
# python3 examples/RF/theories-types-file.py -dataset=examples/datasets_converted/australian_0.csv -types=examples/datasets_converted/australian_0.types
# Check V1.0: Ok 
# Check V2.0: Australian dataset not available

# Machine learning part
learner = Learning.Xgboost(Tools.Options.dataset, problem_type='classification', instances_type='tabular', labels_type='classes')
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT)
instances = learner.get_instances(model, n=10)

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

explainer = Explaining.initialize(model, features_type=australian_types)
for (instance, prediction) in instances:
    print("prediction:", prediction)
    explainer.set_instance(instance)

    tree_specific_reason = explainer.minimal_tree_specific_reasons(n=2, time_limit=3)
    print("tree specific reasons:", tree_specific_reason)
    features = explainer.to_features(tree_specific_reason[0])

    #print("is implicant BT:", explainer.is_tree_specific_reason(tree_specific_reason))
    #print("features tree specific:", features)

