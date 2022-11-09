from pyxai import Builder, Explainer

node1_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=-0.2, right=0.3)
node1_2 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=-0.3, right=node1_1)
node1_3 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=0.4, right=node1_2)
node1_4 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.5, right=node1_3)
tree1 = Builder.DecisionTree(4, node1_4)

node2_1 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.4, right=0.3)
node2_2 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=-0.2, right=node2_1)
node2_3 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=node2_2, right=0.5)
tree2 = Builder.DecisionTree(4, node2_3)

node3_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=0.2, right=0.3)

node3_2_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2, left=-0.2, right=0.2)

node3_2_2 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.1, right=node3_1)
node3_2_3 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-0.5, right=0.1)

node3_3_1 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=node3_2_1, right=node3_2_2)
node3_3_2 = Builder.DecisionNode(2, operator=Builder.GT, threshold=1, left=-0.4, right=node3_2_3)

node3_4 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=node3_3_1, right=node3_3_2)

tree3 = Builder.DecisionTree(4, node3_4)

BTs = Builder.BoostedTrees([tree1, tree2, tree3], n_classes=2)

instance = (4, 3, 1, 1)
print("instance:", instance)

explainer = Explainer.initialize(BTs, instance)

print("target_prediction:", explainer.target_prediction)
print("implicant:", explainer.binary_representation)

print("---------------------------------------------------")
direct = explainer.direct_reason()
print("direct reason:", direct)
direct_features = explainer.to_features(direct)
print("to_features:", direct_features)
assert direct_features == ('f4 == 1', 'f2 > 1', 'f3 == 1', 'f1 > 2'), "The direct reason is not correct."

print("---------------------------------------------------")
tree_specific = explainer.tree_specific_reason()
print("tree specific reason:", tree_specific)
tree_specific_feature = explainer.to_features(tree_specific)
print("to_features:", tree_specific_feature)
print("is_tree_specific:", explainer.is_tree_specific_reason(tree_specific))
print("is_sufficient_reason:", explainer.is_sufficient_reason(tree_specific))

print("---------------------------------------------------")
sufficient = explainer.sufficient_reason()
print("sufficient reason:", sufficient)
sufficient_feature = explainer.to_features(sufficient)
print("to_features:", sufficient_feature)
print("is_tree_specific:", explainer.is_tree_specific_reason(sufficient))
print("is_sufficient_reason:", explainer.is_sufficient_reason(sufficient))

print("---------------------------------------------------")
reason = (1, 4)
feat = explainer.to_features(reason)
print("feat:", feat)
print("is_tree_specific:", explainer.is_tree_specific_reason(reason))
print("is_sufficient_reason:", explainer.is_sufficient_reason(reason))

print("---------------------------------------------------")
reason = (2, 1)
feat = explainer.to_features(reason)
print("feat:", feat)
print("is_tree_specific:", explainer.is_tree_specific_reason(reason))
print("is_sufficient_reason:", explainer.is_sufficient_reason(reason))
