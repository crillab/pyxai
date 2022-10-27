import sys

from pyxai import Learning, Explainer, Tools

# Machine learning part


# usage
# python3 pyxai/examples/DT/Simple.py -dataset=path/to/dataset.csv
learner = Learning.Scikitlearn(Tools.Options.dataset)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
instance, prediction = learner.get_instances(model, n=1, correct=True)

# Explanation part
explainer = Explainer.decision_tree(model, instance)
print("instance:", instance)

if len(explainer.binary_representation) < 15:
    print("binary representation: ", explainer.binary_representation)
print("prediction:", prediction)
print("len binary representation", len(explainer.binary_representation))

direct_reason = explainer.direct_reason()
print("\nlen direct:", len(direct_reason))
print("is a reason:", explainer.is_reason(direct_reason))

sufficient_reason = explainer.sufficient_reason(n=1)
print("\nlen sufficient reason:", len(sufficient_reason))
print("to features", explainer.to_features(sufficient_reason))
print("is sufficient_reason (for max 50 checks): ", explainer.is_sufficient_reason(sufficient_reason, n_samples=50))
print()
minimal = explainer.minimal_sufficient_reason()
print("\nminimal:", len(minimal))
print("is sufficient_reason (for max 50 checks): ", explainer.is_sufficient_reason(sufficient_reason, n_samples=50))

print("\nnecessary literals: ", explainer.necessary_literals())
print("\nrelevant literals: ", explainer.relevant_literals())

sufficient_reasons_per_attribute = explainer.n_sufficient_reasons_per_attribute()
print("\nsufficient_reasons_per_attribute:", sufficient_reasons_per_attribute)

constractive_reasons = explainer.contrastive_reason(n=Explainer.ALL)
print("\nnb constractive_reasons:", len(constractive_reasons))

all_are_contrastive = True
for contrastive in constractive_reasons:
    if not explainer.is_contrastive_reason(contrastive):
        print(f"{contrastive} is not a contrastive reason")
        all_are_contrastive = False

if all_are_contrastive:
    print("All contrastive are ok")
