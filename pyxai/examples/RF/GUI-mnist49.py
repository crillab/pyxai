from time import time
import numpy
from pyxai import Learning, Explaining, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv

# Check V1.0: Ok 
# Check V2.0: Ok except AttributeError: type object 'Qt' has no attribute 'OpenHandCursor'


# the location of the dataset
#dataset = "../datasets_ijcai/mnist49.csv" # Dataset not available
dataset = 'examples/Converters/mnist/misnt_3_8.csv'

# Machine learning part
learner = Learning.Scikitlearn(dataset, problem_type=Learning.CLASSIFICATION)
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.RF)
instances = learner.get_instances(model, n=10, is_correct=True)

# Explanation part
explainer = Explaining.initialize(model)

for (instance, prediction) in instances:
    explainer.set_instance(instance)
    direct = explainer.direct_reason()
    print("len direct:", len(direct))

    majoritary_reason = explainer.majoritary_reason(n=1)
    print("len majoritary_reason:", len(majoritary_reason))
    #assert explainer.is_reason(majoritary_reason), "This is not a reason reason !"

    # minimal_majoritary = explainer
    minimal_reason = explainer.minimal_majoritary_reason(time_limit=60)
    #assert explainer.is_reason(minimal_reason), "This is not a reason"
    if explainer.elapsed_time == Explaining.TIMEOUT: print("This is an approximation")
    print("len minimal majoritary reason:", len(minimal_reason))


# Visualization part
def get_pixel_value(instance, x, y, shape):
    index = x * shape[0] + y 
    return instance[index]

def instance_index_to_pixel_position(i, shape):
    return i // shape[0], i % shape[0]

explainer.visualisation.gui(image={"shape": (28,28),
                      "dtype": numpy.uint8,
                      "get_pixel_value": get_pixel_value,
                      "instance_index_to_pixel_position": instance_index_to_pixel_position})

