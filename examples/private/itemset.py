from pyxai import Learning, Explainer, Tools
from efficient_apriori import apriori

# Machine learning part
Tools.set_verbose(0)
learner = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)

model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
instances = learner.get_instances(model, n=20)
explainer = Explainer.decision_tree(model)

transactions = []
for tuple in instances:
    instance = tuple[0]
    explainer.set_instance(instance)
    transactions.append(explainer.binary_representation)

itemsets, rules = apriori(transactions, min_support=0.5, min_confidence=1)

print(rules)
