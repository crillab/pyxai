# Check V1.0: Ok 
# Check V1.0: Dataset not available

from pyxai import Learning, Explaining

# Machine learning part
learner = Learning.Scikitlearn("tests/datasets/australian_0.csv", problem_type=Learning.CLASSIFICATION) # Dataset deketed
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF)
instances = learner.get_instances(model, n=5, seed=11200, is_correct=True)

australian_types = {
    "numerical": Learning.DEFAULT,
    "categorical": {"A4*": (1, 2, 3), 
                    "A5*": tuple(range(1, 15)),
                    "A6*": (1, 2, 3, 4, 5, 7, 8, 9), 
                    "A12*": tuple(range(1, 4))},
    "binary": ["A1", "A8", "A9", "A11"],
}

# Explainer part

explainer = Explaining.initialize(model, features_type=australian_types)
for (instance, prediction) in instances:
    explainer.set_instance(instance)
    majoritary_reason = explainer.majoritary_reason(time_limit=10)
    print("majoritary_reason", len(majoritary_reason))
    #majoritary_reason = explainer.majoritary_reason(time_limit=50)
    #majoritary_reason = explainer.majoritary_reason(time_limit=100)


#print("\nlen tree_specific: ", len(majoritary_reason))

#print("\ntree_specific without intervales: ", explainer.to_features(majoritary_reason, without_intervals=True))
#print("\ntree_specific: ", explainer.to_features(majoritary_reason))
#print("is majoritary:", explainer.is_majoritary_reason(majoritary_reason))

explainer.visualisation.gui()