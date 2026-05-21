import numpy
from pyxai import Learning, Explaining, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv

# Check V1.0: Ok
# Check V2.0: Ok

# the location of the dataset
path = ""
dataset = "examples/Converters/mnist/misnt_3_8.csv"

# Machine learning part
learner = Learning.Xgboost(dataset, problem_type='classification', instances_type='tabular', labels_type='classes')
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type='boosted-tree')
instances = learner.get_instances(model, n=2, is_correct=True, subset_predicted_classes=[0])

# Explanation part
explainer = Explaining.initialize(model)
for (instance, prediction) in instances:
    explainer.set_instance(instance)

    direct = explainer.direct_reason()
    print("len direct:", len(direct))

    tree_specific_reason = explainer.tree_specific_reason()
    print("len tree_specific_reason:", len(tree_specific_reason))

    #minimal_tree_specific_reason = explainer.minimal_tree_specific_reason(time_limit=20)
    #print("len minimal tree_specific_reason:", len(minimal_tree_specific_reason))

def get_pixel_value(instance, x, y, shape):
    index = x * shape[0] + y 
    return instance[index]

def instance_index_to_pixel_position(i, shape):
    return i // shape[0], i % shape[0]

explainer.visualisation.screen(instance, minimal_tree_specific_reason,
                      image={"shape": (28,28),
                      "dtype": numpy.uint8,
                      "get_pixel_value": get_pixel_value,
                      "instance_index_to_pixel_position": instance_index_to_pixel_position})
    