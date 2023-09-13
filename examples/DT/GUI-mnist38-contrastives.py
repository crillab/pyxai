from pyxai import Learning, Explainer, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv
# Check V1.0: Ok

# the location of the dataset
import numpy
dataset = "./examples/datasets_not_converted/mnist38.csv"

# Machine learning part
learner = Learning.Scikitlearn(dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
instances = learner.get_instances(model, n=10, correct=False)

# Explanation part
explainer = Explainer.initialize(model)

for (instance, prediction) in instances:
    explainer.set_instance(instance)
    contrastive_reasons = explainer.contrastive_reason(n=Explainer.ALL)

    # Visualisation part
    explainer.heat_map("heat map 1", contrastive_reasons, contrastive=True)

def get_pixel_value(instance, x, y, shape):
    index = x * shape[0] + y 
    return instance[index]

def instance_index_to_pixel_position(i, shape):
    return i // shape[0], i % shape[0]

explainer.show(image={"shape": (28,28),
                      "dtype": numpy.uint8,
                      "get_pixel_value": get_pixel_value,
                      "instance_index_to_pixel_position": instance_index_to_pixel_position})

