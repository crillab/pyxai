from time import time

from pyxai import Learning, Explainer, Tools

# To use with the minist dataset for example
# (classification between 4 and 9 or between 3 and 8)
# available here:
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist38.csv
# http://www.cril.univ-artois.fr/expekctation/datasets/mnist49.csv


# the location of the dataset
path = ""
dataset = "../../data/dataML/mnist38.csv"

# Machine learning part
learner = Learning.Xgboost(dataset, learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT)
instances = learner.get_instances(model, n=3, correct=True)

# Explanation part
explainer = Explainer.initialize(model)
for (instance, prediction) in instances:
    explainer.set_instance(instance)

    direct = explainer.direct_reason()
    print("len direct:", len(direct))

    tree_specific_reason = explainer.tree_specific_reason()
    print("len tree_specific_reason:", len(tree_specific_reason))

    minimal_tree_specific_reason = explainer.minimal_tree_specific_reason(time_limit=5)
    print("len minimal tree_specific_reason:", len(minimal_tree_specific_reason))


explainer.show(image=(28, 28))
