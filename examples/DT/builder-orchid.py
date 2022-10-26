# Example taken from the paper quoted below: "The decision tree in Figure 1 separates Cattleya orchids from other
# orchids using the following features: x1: “has fragrant flowers”, x2: “has one or two leaves”,
# x3: “has large flowers”, and x4: “is sympodial”."

# @article{AUDEMARD2022102088,
# title = {On the explanatory power of Boolean decision trees},
# journal = {Data & Knowledge Engineering},
# pages = {102088},
# year = {2022},
# issn = {0169-023X},
# doi = {https://doi.org/10.1016/j.datak.2022.102088},
# url = {https://www.sciencedirect.com/science/article/pii/S0169023X22000799},
# author = {Gilles Audemard and Steve Bellart and Louenas Bounia and Frédéric Koriche and Jean-Marie Lagniez and Pierre Marquis},
# }

from pyxai import Builder, Explainer

# Builder part
node_x4_1 = Builder.DecisionNode(4, left=0, right=1)
node_x4_2 = Builder.DecisionNode(4, left=0, right=1)
node_x4_3 = Builder.DecisionNode(4, left=0, right=1)
node_x4_4 = Builder.DecisionNode(4, left=0, right=1)
node_x4_5 = Builder.DecisionNode(4, left=0, right=1)

node_x3_1 = Builder.DecisionNode(3, left=0, right=node_x4_1)
node_x3_2 = Builder.DecisionNode(3, left=node_x4_2, right=node_x4_3)
node_x3_3 = Builder.DecisionNode(3, left=node_x4_4, right=node_x4_5)

node_x2_1 = Builder.DecisionNode(2, left=0, right=node_x3_1)
node_x2_2 = Builder.DecisionNode(2, left=node_x3_2, right=node_x3_3)

node_x1_1 = Builder.DecisionNode(1, left=node_x2_1, right=node_x2_2)

tree = Builder.DecisionTree(4, node_x1_1, force_features_equal_to_binaries=True)

# Explainer part for instance = (1,1,1,1)
print("instance = (1,1,1,1):")
explainer = Explainer.initialize(tree, instance=(1, 1, 1, 1))

print("target_prediction:", explainer.target_prediction)
direct = explainer.direct_reason()
print("direct:", direct)
assert direct == (1, 2, 3, 4), "The direct reason is not good !"

sufficient_reasons = explainer.sufficient_reason(n=Explainer.ALL)
print("sufficient_reasons:", sufficient_reasons)
assert sufficient_reasons == ((1, 4), (2, 3, 4)), "The sufficient reasons are not good !"

for sufficient in sufficient_reasons:
    assert explainer.is_sufficient_reason(sufficient), "This is have to be a sufficient reason !"

minimals = explainer.minimal_sufficient_reason()
print("Minimal sufficient reasons:", minimals)
assert minimals == (1, 4), "The minimal sufficient reasons are not good !"

contrastives = explainer.contrastive_reason(n=Explainer.ALL)
print("Contrastives:", contrastives)
for contrastive in contrastives:
    assert explainer.is_contrastive_reason(contrastive), "This is not a contrastive reason !"

# Explainer part for instance = (0,0,0,0)
print("\ninstance = (0,0,0,0):")

explainer.set_instance((0, 0, 0, 0))

print("target_prediction:", explainer.target_prediction)
direct = explainer.direct_reason()
print("direct:", direct)
assert direct == (-1, -2), "The direct reason is not good !"

sufficient_reasons = explainer.sufficient_reason(n=Explainer.ALL)
print("sufficient_reasons:", sufficient_reasons)
assert sufficient_reasons == ((-4,), (-1, -2), (-1, -3)), "The sufficient reasons are not good !"
for sufficient in sufficient_reasons:
    assert explainer.is_sufficient_reason(sufficient), "This is not a sufficient reason !"
minimals = explainer.minimal_sufficient_reason(n=1)
print("Minimal sufficient reasons:", minimals)
assert minimals == (-4,), "The minimal sufficient reasons are not good !"
