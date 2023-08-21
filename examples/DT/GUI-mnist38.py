from pyxai import Learning, Explainer, Tools
import numpy
# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv
# Check V1.0: Ok

# the location of the dataset
dataset = "./examples/datasets_not_converted/mnist38.csv"

# Machine learning part
learner = Learning.Scikitlearn(dataset, learner_type=Learning.CLASSIFICATION)
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
print("\nnumber of sufficient reasons:", explainer.n_sufficient_reasons())

def get_pixel_value(instance, x, y, shape):
    index = x * shape[0] + y 
    return instance[index]

def instance_index_to_pixel_position(i, shape):
    return i // shape[0], i % shape[0]

# Visualization part
explainer.heat_map("heat map 1", sufficient_reasons_per_attribute)
explainer.show(image={"shape": (28,28),
                      "dtype": numpy.uint8,
                      "get_pixel_value": get_pixel_value,
                      "instance_index_to_pixel_position": instance_index_to_pixel_position})
