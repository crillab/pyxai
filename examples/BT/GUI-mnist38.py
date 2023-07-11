import numpy
from pyxai import Learning, Explainer, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv

# Check V1.0: Ok

# the location of the dataset
path = ""
dataset = "../../data/dataML/mnist38.csv"

# Machine learning part
learner = Learning.Xgboost(dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT)
instances = learner.get_instances(model, n=10, correct=True, predictions=[0])

# Explanation part
explainer = Explainer.initialize(model)
for (instance, prediction) in instances:
    explainer.set_instance(instance)

    direct = explainer.direct_reason()
    print("len direct:", len(direct))

    tree_specific_reason = explainer.tree_specific_reason()
    print("len tree_specific_reason:", len(tree_specific_reason))

    minimal_tree_specific_reason = explainer.minimal_tree_specific_reason(time_limit=100)
    print("len minimal tree_specific_reason:", len(minimal_tree_specific_reason))

def get_pixel_value(instance, x, y, shape):
    index = x * shape[0] + y 
    return instance[index]

def instance_index_to_pixel_position(i, shape):
    return i // shape[0], i % shape[0]

explainer.show(image={"shape": (28,28),
                      "dtype": numpy.uint8,
                      "get_pixel_value": get_pixel_value,
                      "instance_index_to_pixel_position": instance_index_to_pixel_position})