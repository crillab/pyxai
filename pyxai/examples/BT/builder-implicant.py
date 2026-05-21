# Boosted Trees from the paper: Computing Abductive Explanations for Boosted Trees
# https://arxiv.org/pdf/2209.07740.pdf

# Check V1.0: Ok
# Check V2.0: OK
# Note: Sufficient reason for BT have to be implemented in another way

from pyxai import Builder, Explaining

node1_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=-0.2, right=0.3)
node1_2 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=-0.3, right=node1_1)
node1_3 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=0.4, right=node1_2)
node1_4 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.5, right=node1_3)
tree1 = Builder.DecisionTree(4, node1_4, force_features_equal_to_binaries=True)

node2_1 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.4, right=0.3)
node2_2 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=-0.2, right=node2_1)
node2_3 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=node2_2, right=0.5)
tree2 = Builder.DecisionTree(4, node2_3, force_features_equal_to_binaries=True)

node3_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=0.2, right=0.3)

node3_2_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=-0.2, right=0.2)

node3_2_2 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.1, right=node3_1)
node3_2_3 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.5, right=0.1)

node3_3_1 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=node3_2_1, right=node3_2_2)
node3_3_2 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=-0.4, right=node3_2_3)

node3_4 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=node3_3_1, right=node3_3_2)

tree3 = Builder.DecisionTree(4, node3_4, force_features_equal_to_binaries=True)

BTs = Builder.BoostedTrees([tree1, tree2, tree3], n_classes=2)

instance = (4, 3, 1, 1)
print("instance:", instance)

explainer = Explaining.initialize(BTs, instance, features_type={"numerical": ["f1","f2","f4"], "binary": ["f3"]})

print("target_prediction:", explainer.target_prediction)
print("implicant:", explainer.binary_representation)

print("is implicant?:", explainer.is_implicant_BT((1, )))
print("is implicant?:", explainer.is_implicant_BT((2, )))
print("is implicant?:", explainer.is_implicant_BT((3,)))
print("is implicant?:", explainer.is_implicant_BT((4, )))

print("is implicant?:", explainer.is_implicant_BT((1, 1)))


exit(0)
print(" Direct ---------------------------------------------------")
direct = explainer.direct_reason()
print("direct reason:", direct)
direct_features = explainer.to_features(direct)
print("to_features:", direct_features)
#assert direct_features == ('f1 > 2', 'f2 > 1', 'f3 == 1', 'f4 == 1'), "The direct reason is not correct."

print(" Tree Specific ---------------------------------------------------")
tree_specific = explainer.tree_specific_reason()
print("tree specific reason:", tree_specific)
tree_specific_feature = explainer.to_features(tree_specific)
print("to_features:", tree_specific_feature)

print(" Is Implicant ---------------------------------------------------")
# Example from the page: https://www.cril.univ-artois.fr/pyxai/documentation/classification/BTexplanations/sufficient/

explanation = tree_specific 
explanation_feature = explainer.to_features(explanation)
print("explanation features:", explanation_feature)

