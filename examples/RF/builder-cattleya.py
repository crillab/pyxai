# Example taken from the paper quoted below: "Example 1. The random forest F = {T1, T2, T3} in Figure 1 is composed of three decision trees. It separates Cattleya orchids from other orchids using the following features: x1: “has fragrant flowers”, x2: “has one or two leaves”, x3: “has large flowers”, and x4: “is sympodial”."
#
# @inproceedings{DBLP:conf/aaai/AudemardBBKLM22,
#   author    = {Gilles Audemard and Steve Bellart and Louenas Bounia and Fr{\'{e}}d{\'{e}}ric Koriche andJean{-}Marie Lagniez and Pierre Marquis},
#   title     = {Trading Complexity for Sparsity in Random Forest Explanations},
#   booktitle = {Thirty-Sixth {AAAI} Conference on Artificial Intelligence},
#   pages     = {5461--5469},
#   publisher = {{AAAI} Press},
#   year      = {2022},
#   url       = {https://ojs.aaai.org/index.php/AAAI/article/view/20484},
# }


from pyxai import Builder, Explainer

nodeT1_1 = Builder.DecisionNode(1, left=0, right=1)
nodeT1_3 = Builder.DecisionNode(3, left=0, right=nodeT1_1)
nodeT1_2 = Builder.DecisionNode(2, left=1, right=nodeT1_3)
nodeT1_4 = Builder.DecisionNode(4, left=0, right=nodeT1_2)

tree1 = Builder.DecisionTree(4, nodeT1_4, force_features_equal_to_binaries=True)

nodeT2_4 = Builder.DecisionNode(4, left=0, right=1)
nodeT2_1 = Builder.DecisionNode(1, left=0, right=nodeT2_4)
nodeT2_2 = Builder.DecisionNode(2, left=nodeT2_1, right=1)

tree2 = Builder.DecisionTree(4, nodeT2_2, force_features_equal_to_binaries=True)  # 4 features but only 3 used

nodeT3_1_1 = Builder.DecisionNode(1, left=0, right=1)
nodeT3_1_2 = Builder.DecisionNode(1, left=0, right=1)
nodeT3_4_1 = Builder.DecisionNode(4, left=0, right=nodeT3_1_1)
nodeT3_4_2 = Builder.DecisionNode(4, left=0, right=1)

nodeT3_2_1 = Builder.DecisionNode(2, left=nodeT3_1_2, right=nodeT3_4_1)
nodeT3_2_2 = Builder.DecisionNode(2, left=0, right=nodeT3_4_2)

nodeT3_3_1 = Builder.DecisionNode(3, left=nodeT3_2_1, right=nodeT3_2_2)

tree3 = Builder.DecisionTree(4, nodeT3_3_1, force_features_equal_to_binaries=True)

forest = Builder.RandomForest([tree1, tree2, tree3], n_classes=2)
# For instance = (1,1,1,1)
print("For instance = (1,1,1,1):")
print("")
instance = (1, 1, 1, 1)
explainer = Explainer.initialize(forest, instance=instance)
print("target_prediction:", explainer.target_prediction)

direct = explainer.direct_reason()
print("direct:", direct)
assert direct == (1, 2, 3, 4), "The direct reason is not good !"

sufficient = explainer.sufficient_reason()
print("sufficient:", sufficient)
assert explainer.is_sufficient_reason(sufficient)
assert sufficient == (1, 4), "The sufficient reason is not good !"

minimal = explainer.minimal_sufficient_reason()
print("minimal:", minimal)
assert minimal == (1, 4), "The minimal reason is not good !"

majoritary = explainer.majoritary_reason()
print("majoritary:", majoritary)

minimal_contrastives = explainer.minimal_contrastive_reason(n=Explainer.ALL)
print("minimal_contrastive: ", minimal_contrastives)

minimals = explainer.preferred_majoritary_reason(method=Explainer.MINIMAL, n=10)
print("minimals:", minimals)

for c in minimal_contrastives:
    assert explainer.is_contrastive_reason(c), "..."

# For instance = (0,1,0,0)
print("\nFor instance = (0,1,0,0):")
print("")
instance = (0, 1, 0, 0)
explainer.set_instance(instance=instance)
print("target_prediction:", explainer.target_prediction)


direct = explainer.direct_reason()
print("direct:", direct)
assert direct == (2, -3, -4), "The direct reason is not good !"

sufficient = explainer.sufficient_reason()
print("sufficient:", sufficient)
assert sufficient == (-1, -3), "The sufficient reason is not good !"

minimal = explainer.minimal_sufficient_reason()
print("minimal:", minimal)
assert minimal == (-4,), "The minimal reason is not good !"

majoritary = explainer.majoritary_reason(n=Explainer.ALL)
print("majoritary:", majoritary)

minimals = explainer.preferred_majoritary_reason(method=Explainer.MINIMAL, n=10)
print("minimals:", minimals)

minimal_contrastives = explainer.minimal_contrastive_reason(n=Explainer.ALL)
print("minimal_contrastive: ", minimal_contrastives)

for c in minimal_contrastives:
    assert explainer.is_contrastive_reason(c), f"{c} is not a contrastive reason"
