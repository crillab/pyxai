# Example taken from the paper quoted below.

# @misc{https://doi.org/10.48550/arxiv.2206.08758,
#   doi = {10.48550/ARXIV.2206.08758},
#   url = {https://arxiv.org/abs/2206.08758},
#   author = {Coste-Marquis, Sylvie and Marquis, Pierre},
#   keywords = {Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
#   title = {Rectifying Mono-Label Boolean Classifiers},
#   publisher = {arXiv},
#   year = {2022},  
#   copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
# }

from pyxai import Learning, Explainer, Tools

# Machine learning part
learner = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)

print("type:", type(model))
instances = learner.get_instances(model=model)
for (instance, prediction_classifier) in instances:
    prediction_model_1 = model.predict_instance(instance)
    implicant = model.instance_to_binaries(instance)
    prediction_model_2 = model.predict_implicant(implicant)
    
    assert prediction_classifier == prediction_model_1, "problem"
    assert prediction_classifier == prediction_model_2, 'problem'

# Explanation part
explainer = Explainer.initialize(model, instance)

tree1 = model
tree2 = None
tree3 = None

rectifying_tree = explainer.rectify(tree1, tree2, tree3)


print("Original tree:", model.raw_data_for_CPP())


#print("Positive rectifying tree:", positive_rectifying_tree.raw_data_for_CPP())
#print("Negative rectifying tree:", negative_rectifying_tree.raw_data_for_CPP())
#print("Resulting tree:", rectifying_tree.raw_data_for_CPP())
