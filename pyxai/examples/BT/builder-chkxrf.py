from pyxai import Builder, Learning, Explaining
import random as rand


#Tree 1 
tree_1_x_2 = Builder.DecisionNode(2, operator=Builder.GE, threshold=0.817999959, left=0.0290748924, right=-0.33644861)

tree_1_x_3 = Builder.DecisionNode(3, operator=Builder.GE, threshold=1.449, left=-0.541057169, right=0.120000005)

tree_1_x_1 = Builder.DecisionNode(1, operator=Builder.GE, threshold=0.277500004, left=-0.0100278556
, right=0.226630449)

tree_1_x_3_bis = Builder.DecisionNode(3, operator=Builder.GE, threshold=1.02250004, left=-0.564179122
, right=-0.0461538509)


tree_1_x_4 = Builder.DecisionNode(4, operator=Builder.GE, threshold=-0.308499992, left=tree_1_x_2, right=tree_1_x_3)

tree_1_x_1_bis = Builder.DecisionNode(1, operator=Builder.GE, threshold=1.403, left=tree_1_x_1, right=tree_1_x_3_bis)

tree_1_root = Builder.DecisionNode(4, operator=Builder.GE, threshold=0.631000042, left=tree_1_x_4, right=tree_1_x_1_bis)
tree_1 = Builder.DecisionTree(5, tree_1_root)

#tree 2

tree_2_x_1 = Builder.DecisionNode(1, operator=Builder.GE, threshold=0.172499999, left=-0.153566822, right=-0.456195921)

tree_2_x_5 = Builder.DecisionNode(5, operator=Builder.GE, threshold=0.621999979, left=-0.241189703, right=0.352639019)

tree_2_x_1_bis = Builder.DecisionNode(1, operator=Builder.GE, threshold=0.159500003, left=-0.0891945809
, right=0.154747352)

tree_2_x_3 = Builder.DecisionNode(3, operator=Builder.GE, threshold=1.31799996, left=-0.421002328
, right=0.0654052421)

tree_2_x_3_bis = Builder.DecisionNode(3, operator=Builder.GE, threshold=0.685000002, left=tree_2_x_1, right=tree_2_x_5)

tree_2_x_1_bis_bis = Builder.DecisionNode(1, operator=Builder.GE, threshold=1.324, left=tree_2_x_1_bis, right=tree_2_x_3)

tree_2_root = Builder.DecisionNode(4, operator=Builder.GE, threshold=0.702499986, left=tree_2_x_3_bis, right=tree_2_x_1_bis_bis)

tree_2 = Builder.DecisionTree(5, tree_2_root)



learner_information = Learning.LearnerInformation(None, problem_type='classification')
learner_information.extras["base_score"] = 0.0
BTs = Builder.BoostedTrees([tree_1, tree_2], n_classes=2, feature_names=["x1", "x2", "x3", "x4", "x5"], learner_information=learner_information)


print("x = 3.306, 0.653, 0.313, 0.669, -0.218:")
instance = (3.306, 0.653, 0.313, 0.669, -0.218)

explainer = Explaining.initialize(BTs, features_type={"numerical": ["x1", "x2", "x3", "x4", "x5"]})
#explainer = Explaining.initialize(BTs)

explainer.set_instance(instance=instance)

print("intance:", instance)
print("binary representation: ", explainer.binary_representation)
print("target_prediction:", explainer.target_prediction)

print("instance to_features:", explainer.to_features(explainer.binary_representation))

contrastive = explainer.minimal_contrastive_reason()
print("contrastive:", explainer.to_features(contrastive))

