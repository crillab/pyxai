from pyxai import Learning, Explaining, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv
# Check V1.0: Ok
# Check V2.0: Ok


# the location of the dataset
import numpy
dataset = "./examples/Converters/mnist/misnt_3_8.csv"

# Machine learning part
learner = Learning.Scikitlearn(dataset, problem_type=Learning.CLASSIFICATION, instances_type='tabular')
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.DT)
instances = learner.get_instances(model, n=10, is_correct=False)

# Explanation part
explainer = Explaining.initialize(model)

for (instance, prediction) in instances:
    explainer.set_instance(instance)
    contrastive_reasons = explainer.contrastive_reason(n=Explaining.ALL)

    # Visualisation part
    explainer.visualisation.heat_map("heat map 1", contrastive_reasons, contrastive=True)

def get_pixel_value(instance, x, y, shape):
    index = x * shape[0] + y 
    return instance[index]

def instance_index_to_pixel_position(i, shape):
    return i // shape[0], i % shape[0]

explainer.visualisation.gui(image={"shape": (28,28),
                      "dtype": numpy.uint8,
                      "get_pixel_value": get_pixel_value,
                      "instance_index_to_pixel_position": instance_index_to_pixel_position})

