from time import time

from pyxai import Learning, Explainer, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv


# the location of the dataset
dataset = "./examples/datasets/mnist49.csv"

# Machine learning part
learner = Learning.Scikitlearn(dataset)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
instance, prediction = learner.get_instances(model, n=1, correct=True)

print("instance:", instance)
print("prediction:", prediction)

# Explanation part
explainer = Explainer.initialize(model, instance)
direct = explainer.direct_reason()
print("direct:", direct)

sufficient_reasons = explainer.sufficient_reason(n=100, time_limit=100)  # take 100 in order to have different reasons
assert explainer.is_sufficient_reason(sufficient_reasons[-1]), "This is not a sufficient reason !"

minimal = explainer.minimal_sufficient_reason()
print("minimal:", minimal)
assert explainer.is_sufficient_reason(minimal), "This is not a sufficient reason !"

sufficient_reasons_per_attribute = explainer.n_sufficient_reasons_per_attribute()
print("\nsufficient_reasons_per_attribute:", sufficient_reasons_per_attribute)

# Visualization part
vizualisation = Tools.Vizualisation(28, 28, instance)

image1 = vizualisation.new_image("Instance").set_instance(instance)

image2 = vizualisation.new_image("Minimal")
image2.add_reason(explainer.to_features(minimal, details=True))
image2.set_background_instance(instance)

image3 = vizualisation.new_image("Sufficient")
image3.add_reason(explainer.to_features(sufficient_reasons[-1], details=True))
image3.set_background_instance(instance)

image3 = vizualisation.new_image("heatmap")
image3.add_reason(explainer.to_features(sufficient_reasons_per_attribute, details=True))
image3.set_background_instance(instance)

vizualisation.display(n_rows=2)
