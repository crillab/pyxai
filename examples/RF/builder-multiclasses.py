# Check V1.0: Ok but minimal_majoritary_reason() return () problem

from pyxai import Builder, Explainer

node_1_1 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=3, right=2)
node_1_2 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=1, right=node_1_1)
node_1_3 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=1, right=node_1_2)
node_1_4 = Builder.DecisionNode(1, operator=Builder.EQ, threshold=1, left=3, right=node_1_3)
tree_1 = Builder.DecisionTree(6, node_1_4)

node_2_1 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=1, right=2)
node_2_2 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=3, right=node_2_1)
node_2_3 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=node_2_2, right=3)
tree_2 = Builder.DecisionTree(6, node_2_3)

node_3_1 = Builder.DecisionNode(1, operator=Builder.EQ, threshold=1, left=3, right=1)
node_3_2 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=2, right=4)
node_3_3 = Builder.DecisionNode(5, operator=Builder.EQ, threshold=1, left=2, right=node_3_2)
node_3_4 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=node_3_1, right=node_3_3)
tree_3 = Builder.DecisionTree(6, node_3_4)

node_4_1 = Builder.DecisionNode(6, operator=Builder.EQ, threshold=1, left=4, right=3)
tree_4 = Builder.DecisionTree(6, node_4_1)

node_5_1 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=2, right=4)
node_5_2 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=3, right=node_5_1)
node_5_3 = Builder.DecisionNode(1, operator=Builder.EQ, threshold=1, left=1, right=node_5_2)
tree_5 = Builder.DecisionTree(6, node_5_3)

forest = Builder.RandomForest([tree_1, tree_2, tree_3, tree_4, tree_5], n_classes=5)

instance = [0, 0, 1, 1, 0, 0]
instance = [0, 0, 0, 0, 0, 0]
instance = [1, 0, 0, 0, 1, 1]
explainer = Explainer.initialize(forest, instance=instance)
print("instance", instance)
print("binary", explainer.binary_representation)
print("to f", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))
print("prediction: ", explainer.target_prediction)

print("direct     reason: ", explainer.direct_reason())
print("majoritary reason: ", explainer.majoritary_reason(seed=6, n_iterations=1))
minimal_majoritary_reason = explainer.minimal_majoritary_reason()
print("minimal majoritary reason: ", minimal_majoritary_reason)
print("is majoritary", explainer.is_majoritary_reason(minimal_majoritary_reason))
print("sufficient reason: ", explainer.sufficient_reason())

instance = [0, 0, 0, 1, 1, 1]  # the same number of votes :(
explainer.set_instance(instance)
print("instance", instance)
print("prediction: ", explainer.target_prediction)

print("direct_reason: ", explainer.direct_reason())
print("majoritary reason: ", explainer.majoritary_reason(seed=6, n_iterations=1))
minimal_majoritary_reason = explainer.minimal_majoritary_reason(n=1)
print("minimal majoritary reason: ", minimal_majoritary_reason)

if minimal_majoritary_reason is not None:
    print("is majoritary", explainer.is_majoritary_reason(minimal_majoritary_reason))
    print("majoritary reason: ", explainer.majoritary_reason(seed=6, n_iterations=1))

sufficient_reason = explainer.sufficient_reason()
print("sufficient reason: ", sufficient_reason)
