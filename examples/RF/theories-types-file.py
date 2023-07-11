from pyxai import Learning, Explainer, Tools

# usage
# python3 examples/RF/theories-types-file.py -dataset=examples/datasets_converted/australian_0.csv -types=examples/datasets_converted/australian_0.types
# Check V1.0: Ok 

# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
instances = learner.get_instances(n=10)



# Explainer part
#explainer = Explainer.initialize(model, instance=instance, features_type=)

explainer = Explainer.initialize(model, features_type=Tools.Options.types)
for (instance, prediction) in instances:
    explainer.set_instance(instance)
    

#contrastive = explainer.minimal_contrastive_reason(time_limit=100)
#features = explainer.to_features(contrastive, contrastive=True)

#print("contrastive:", contrastive)
#print("features contrastive:", features)

    majoritary_reason = explainer.majoritary_reason(n_iterations=10)
    print("10")
    majoritary_reason = explainer.majoritary_reason(n_iterations=100)
    print("100")
    majoritary_reason = explainer.majoritary_reason(n_iterations=500)
    print("500")   
    features = explainer.to_features(majoritary_reason)
    #print("features majoritary:", features)

explainer.show()

