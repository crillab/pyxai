from time import time

from pyxai import Learning, Explainer, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv


# the location of the dataset
path = ""
dataset = "./examples/datasets/ mnist49.csv"

# Machine learning part
learner = Learning.Xgboost(dataset)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT)
instance, prediction = learner.get_instances(model, n=1, correct=True)

# Explanation part
explainer = Explainer.initialize(model, instance)
direct = explainer.direct_reason()
print("len direct:", len(direct))

tree_specific_reason = explainer.tree_specific_reason()
print("len tree_specific_reason:", len(tree_specific_reason))
print("is tree specific (for 50 checks)", explainer.is_tree_specific_reason(tree_specific_reason))

minimal_tree_specific_reason = explainer.minimal_tree_specific_reason(time_limit=5)
# assert explainer.is_implicant(minimal_reason), "This is have to be a sufficient reason !"
if (explainer.elapsed_time == Explainer.TIMEOUT): print("this is an approximation of a minimal")
print("len minimal tree specific reason:", len(minimal_tree_specific_reason))
print("is tree specific (for 50 checks)", explainer.is_tree_specific_reason(minimal_tree_specific_reason))

# Heatmap part
vizualisation = Tools.Vizualisation(28, 28, instance)

vizualisation.new_image("Instance").set_instance(instance)
vizualisation.new_image("direct").add_reason(explainer.to_features(direct, details=True)).set_background_instance(instance)
vizualisation.new_image("Tree specific").add_reason(explainer.to_features(tree_specific_reason, details=True)).set_background_instance(instance)
vizualisation.new_image("minimal").add_reason(explainer.to_features(minimal_tree_specific_reason, details=True)).set_background_instance(instance)

vizualisation.display()
