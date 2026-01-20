from pyxai import Builder, Explainer
import random as rand
#Tree 1
tree_1_x_2 = Builder.DecisionNode(2, operator=Builder.GT, threshold=143.5, left=0, right=1)
tree_1_x_8 = Builder.DecisionNode(8, operator=Builder.GT, threshold=40.5, left=0, right=1)
tree_1_x_6 = Builder.DecisionNode(6, operator=Builder.GT, threshold=37.95, left=0, right=1)
tree_1_x_7 = Builder.DecisionNode(7, operator=Builder.GT, threshold=0.411, left=0, right=1)
tree_1_x_8_bis = Builder.DecisionNode(8, operator=Builder.GT, threshold=60.5, left=0, right=1)

tree_1_x_3 = Builder.DecisionNode(3, operator=Builder.GT, threshold=98, left=tree_1_x_2, right=1)
tree_1_x_5 = Builder.DecisionNode(5, operator=Builder.GT, threshold=151.5, left=tree_1_x_8, right=tree_1_x_6)
tree_1_x_2 = Builder.DecisionNode(2, operator=Builder.GT, threshold=128, left=tree_1_x_7, right=tree_1_x_8_bis)

tree_1_x_4 = Builder.DecisionNode(4, operator=Builder.GT, threshold=32.5, left=tree_1_x_3, right=tree_1_x_5)
tree_1_x_3_bis = Builder.DecisionNode(3, operator=Builder.GT, threshold=108, left=tree_1_x_2, right=1)

tree_1_root = Builder.DecisionNode(1, operator=Builder.GT, threshold=6.5, left=tree_1_x_4, right=tree_1_x_3_bis)
tree_1 = Builder.DecisionTree(8, tree_1_root)

#tree 2
tree_2_x_2 = Builder.DecisionNode(2, operator=Builder.GT, threshold=139, left=0, right=1)
tree_2_x_6 = Builder.DecisionNode(6, operator=Builder.GT, threshold=23.5, left=0, right=1)

tree_2_x_8 = Builder.DecisionNode(8, operator=Builder.GT, threshold=60.5, left=tree_2_x_2, right=0)

tree_2_x_2_bis = Builder.DecisionNode(2, operator=Builder.GT, threshold=111.5, left=0, right=tree_2_x_6)

tree_2_x_6_bis = Builder.DecisionNode(6, operator=Builder.GT, threshold=24.7, left=0, right=tree_2_x_8)
tree_2_root = Builder.DecisionNode(1, operator=Builder.GT, threshold=6.5, left=tree_2_x_2_bis, right=tree_2_x_6_bis)
tree_2 = Builder.DecisionTree(8, tree_2_root)

#tree 3
tree_3_x_7 = Builder.DecisionNode(7, operator=Builder.GT, threshold=1.121, left=0, right=1)
tree_3_x_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=7.5, left=0, right=1)
tree_3_x_6 = Builder.DecisionNode(6, operator=Builder.GT, threshold=25.8, left=1, right=0)
tree_3_x_5 = Builder.DecisionNode(5, operator=Builder.GT, threshold=587.5, left=1, right=0)

tree_3_x_8 = Builder.DecisionNode(8, operator=Builder.GT, threshold=29.5, left=tree_3_x_7, right=tree_3_x_1)
tree_3_x_8_bis = Builder.DecisionNode(8, operator=Builder.GT, threshold=37.5, left=0, right=1)
tree_3_x_6_bis = Builder.DecisionNode(6, operator=Builder.GT, threshold=29.85, left=tree_3_x_6, right=tree_3_x_5)

tree_3_x_6_ter = Builder.DecisionNode(6, operator=Builder.GT, threshold=26.75, left=0, right=tree_3_x_8)

tree_3_x_2 = Builder.DecisionNode(2, operator=Builder.GT, threshold=157.5, left=tree_3_x_8_bis, right=tree_3_x_6_bis)

tree_3_root = Builder.DecisionNode(2, operator=Builder.GT, threshold=123.5, left=tree_3_x_6_ter, right=tree_3_x_2)
tree_3 = Builder.DecisionTree(8, tree_3_root)

forest = Builder.RandomForest([tree_1, tree_2, tree_3], n_classes=2, feature_names=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"])

instance = (9, 57, 80, 37, 0, 32.8, 0.096, 41)
print("x =", instance)

explainer = Explainer.initialize(forest)
explainer.set_instance(instance=instance)

print("\nbinary representation")
for l in explainer.binary_representation:
    print("  l=", l, " corresponds to ", explainer.to_features([l]))

print("\n\nContrastive without theory")
print("--------------------------")

print("target_prediction:", explainer.target_prediction)
contrastive = explainer.minimal_contrastive_reason()
print("contrastive:", explainer.to_features(contrastive))
print("Change x1 to a value lower or equal than 6.5 gives a prediction 1")
print("But it does not change the value literal 20 since the theory linking literals is not taken into account.")

print("\n\n")
print("Contrastive with theory")
print("--------------------------")
explainer = Explainer.initialize(forest, features_type={"numerical": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]})
explainer.set_instance(instance=instance)
contrastive = explainer.minimal_contrastive_reason()
print("contrastive:", explainer.to_features(contrastive))
print("Change x7 to a value greater than 0.411 gives a prediction 1")
instance2 =  (9, 57, 80, 37, 0, 32.8, 0.412, 41)
print("\nexample instance: ", instance2)
explainer.set_instance(instance2)
print("target_prediction:", explainer.target_prediction)
print("\n\n")
print("In pyxai, contrastive (and sufficient) reasons are asssociated to feature(s) and condition(s), not only features.\n")