from time import time
import numpy
from pyxai import Learning, Explainer, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv
# Check V1.0: Ok 

# the location of the dataset
dataset = "examples/datasets_not_converted/mnist49.csv"

# Machine learning part
learner = Learning.Scikitlearn(dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
instances = learner.get_instances(model, n=10, correct=True)

# Explanation part
explainer = Explainer.initialize(model)

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
    if explainer.elapsed_time == Explainer.TIMEOUT: print("This is an approximation")
    print("len minimal majoritary reason:", len(minimal_reason))


# Visualization part
def get_pixel_value(instance, x, y, shape):
    index = x * shape[0] + y 
    return instance[index]

def instance_index_to_pixel_position(i, shape):
    return i // shape[0], i % shape[0]

explainer.show(image={"shape": (28,28),
                      "dtype": numpy.uint8,
                      "get_pixel_value": get_pixel_value,
                      "instance_index_to_pixel_position": instance_index_to_pixel_position})

