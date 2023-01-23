# usage
# python3 examples/Converters/converter-melb.py -dataset=../../melb_data.csv

from pyxai import Learning, Explainer, Tools


learner = Learning.Scikitlearn(Tools.Options.dataset)

models = learner.evaluate(method=Learning.LEAVE_ONE_GROUP_OUT, output=Learning.RF)

for i, model in enumerate(models):
    instances = learner.get_instances(model=model, indexes=Learning.MIXED, n=10)
    explainer = Explainer.initialize(model, categorical_features=Tools.Options.types) 
    
    for j, (instance, prediction) in enumerate(instances):
        print("Model:", i)
        print("Instance:", j)
        
        explainer.set_instance(instance)
    
        contrastive = explainer.minimal_contrastive_reason(time_limit=100)
        print("Time:", explainer.elapsed_time)
        #print("to_features:", explainer.to_features(contrastive, eliminate_redundant_features=False))

        print("Total binary variables:", len(explainer.binary_representation))
        print("Binary variables to change:", len(contrastive))

        features = explainer.to_features(contrastive, eliminate_redundant_features=True, inverse=True, details=True)

        #print("Total features:", len(instance))
        print("Total features to change:", len(features))
        print("Total features to change (before conversion):", explainer.count_features_before_converting(features))

        print("Is contrastive?:", explainer.is_contrastive_reason(contrastive))

        print("-----------------------------------------------")
