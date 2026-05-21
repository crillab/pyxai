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
#instance = (0, 0, 1, 1)

print("instance:", instance)

explainer = Explaining.initialize(BTs, instance)

print(" Direct ---------------------------------------------------")
direct = explainer.direct_reason()
print("direct reason:", direct)
direct_features = explainer.to_features(direct)
print("to_features:", direct_features)
#assert direct_features == ('f1 > 2', 'f2 > 1', 'f3 == 1', 'f4 == 1'), "The direct reason is not correct."

print(" Tree Specific ---------------------------------------------------")
tree_specifics = explainer.minimal_tree_specific_reasons(n=1)
print("elapsed time:", explainer.elapsed_time)
print("tree specific reasons:", tree_specifics)
print("to_features:", explainer.to_features(tree_specifics))


# instance = (3, 0, 0, 1)
# explainer.set_instance(instance)
# print("instance:", instance)
# print("prediction:", explainer.target_prediction)
# print("binary:", explainer.binary_representation)

# tree_specifics = explainer.tree_specific_reasons()

# print("tree specific reasons:", tree_specifics)
# print("tree specific reasons features:", explainer.to_features(tree_specifics))

# print("is_tree_specific:", explainer.is_tree_specific_reason(tree_specifics))


#exit(0)

for A1 in [0, 3]:
    for A2 in [0, 2]:
        for A3 in [0, 1]:
            for A4 in [0, 1]:
                explainer.set_instance((A1, A2, A3, A4))
                tree_specifics = explainer.minimal_tree_specific_reasons(n=1)
                print("instance:", (A1, A2, A3, A4))
                print("prediction:", explainer.target_prediction)
                print("tree specific reasons:", tree_specifics)


                print("is_tree_specific:", explainer.is_tree_specific_reason(tree_specifics))
