from time import time

from pyxai import Learning, Explainer, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv


# the location of the dataset

dataset = "./examples/datasets_not_converted/mnist38.csv"

# Machine learning part
learner = Learning.Scikitlearn(dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
instance, prediction = learner.get_instances(model, n=1, correct=False)

print("instance:", instance)
print("prediction:", prediction)

# Explanation part
explainer = Explainer.initialize(model, instance)
contrastive_reasons = explainer.contrastive_reason(n=Explainer.ALL)
implicant = explainer.binary_representation

contrastive_reasons_per_attributes = {}

for c in contrastive_reasons:
    for lit in c:
        if lit not in contrastive_reasons_per_attributes:
            contrastive_reasons_per_attributes[lit] = 1
        else:
            contrastive_reasons_per_attributes[lit] += 1

print("contrastive reasaon per attributes:", contrastive_reasons_per_attributes)

explainer.heat_map("heat map 1", contrastive_reasons_per_attributes, contrastive=True)
explainer.show(image_size=(28, 28))

