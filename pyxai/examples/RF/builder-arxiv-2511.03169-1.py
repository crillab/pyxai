#

from pyxai import Builder, Explainer
import random as rand
#Tree 1
tree_1_x_7 = Builder.DecisionNode(7, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_1_x_9 = Builder.DecisionNode(9, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_1_x_8 = Builder.DecisionNode(8, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_1_x_3 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)

tree_1_x_8_n = Builder.DecisionNode(8, operator=Builder.EQ, threshold=1, left=0, right=tree_1_x_9)
tree_1_x_5_n = Builder.DecisionNode(5, operator=Builder.EQ, threshold=1, left=tree_1_x_8, right=tree_1_x_3)
tree_1_x_2_n = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=tree_1_x_8_n, right=tree_1_x_5_n)

tree_1_x_9_n = Builder.DecisionNode(9, operator=Builder.EQ, threshold=1, left=0, right=tree_1_x_7)
tree_1_x_1_r = Builder.DecisionNode(1, operator=Builder.EQ, threshold=1, left=tree_1_x_9_n, right=tree_1_x_2_n)
tree_1 = Builder.DecisionTree(9, tree_1_x_1_r, force_features_equal_to_binaries=True)

#Tree 2
tree_2_x_3 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_2_x_6_n = Builder.DecisionNode(6, operator=Builder.EQ, threshold=1, left=0, right=tree_2_x_3)
tree_2_x_2_n = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=tree_2_x_6_n)

tree_2_x_2 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_2_x_9_n = Builder.DecisionNode(9, operator=Builder.EQ, threshold=1, left=0, right=tree_2_x_2)

tree_2_x_5 = Builder.DecisionNode(5, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_2_x_6_n = Builder.DecisionNode(6, operator=Builder.EQ, threshold=1, left=tree_2_x_9_n, right=tree_2_x_5)

tree_2_x_7_r = Builder.DecisionNode(7, operator=Builder.EQ, threshold=1, left=tree_2_x_2_n, right=tree_2_x_6_n)
tree_2 = Builder.DecisionTree(9, tree_2_x_7_r, force_features_equal_to_binaries=True)


#Tree 3
tree_3_x_8 = Builder.DecisionNode(8, operator=Builder.EQ, threshold=1, left=1, right=0)
tree_3_x_2_n = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=tree_3_x_8)
tree_3_x_1_n = Builder.DecisionNode(1, operator=Builder.EQ, threshold=1, left=0, right=tree_3_x_2_n)

tree_3_x_6 = Builder.DecisionNode(6, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_3_x_5_n = Builder.DecisionNode(5, operator=Builder.EQ, threshold=1, left=0, right=tree_3_x_6)
tree_3_x_8_bis = Builder.DecisionNode(8, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_3_x_9_n = Builder.DecisionNode(9, operator=Builder.EQ, threshold=1, left=tree_3_x_5_n, right=tree_3_x_8_bis)

tree_3_x_7_r = Builder.DecisionNode(7, operator=Builder.EQ, threshold=1, left=tree_3_x_1_n, right=tree_3_x_9_n)
tree_3 = Builder.DecisionTree(9, tree_3_x_7_r, force_features_equal_to_binaries=True)

#Tree 4

tree_4_x_8 = Builder.DecisionNode(8, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_4_x_3_n = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=tree_4_x_8)
tree_4_x_9_n = Builder.DecisionNode(9, operator=Builder.EQ, threshold=1, left=0, right=tree_4_x_3_n)

tree_4_x_4 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_4_x_4_bis = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_4_x_8_n = Builder.DecisionNode(8, operator=Builder.EQ, threshold=1, left=0, right=tree_4_x_4)
tree_4_x_3_n_bis = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=1, right=tree_4_x_4_bis)
tree_4_x_5_n = Builder.DecisionNode(5, operator=Builder.EQ, threshold=1, left=tree_4_x_8_n, right=tree_4_x_3_n_bis)

tree_4_x_6_r = Builder.DecisionNode(6, operator=Builder.EQ, threshold=1, left=tree_4_x_9_n, right=tree_4_x_5_n)
tree_4 = Builder.DecisionTree(9, tree_4_x_6_r, force_features_equal_to_binaries=True)


forest = Builder.RandomForest([tree_1, tree_2, tree_3, tree_4], n_classes=2, feature_names=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"])

instance = (1, 0, 1, 1, 1, 1, 1, 1, 1)
print("x=", instance)
explainer = Explainer.initialize(forest, features_type={"binary": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"],})
explainer.set_instance(instance=instance)
print("intance:", instance)
print("binary representation: ", explainer.binary_representation)
print("target_prediction:", explainer.target_prediction)
print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))
sufficient = explainer.sufficient_reason()
print("sufficient:", explainer.to_features(sufficient))
# check part
print("Check part:", explainer.is_sufficient_reason(sufficient, n_samples=10000))

