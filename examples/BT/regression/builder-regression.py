# Simple regression tree based on paper Computing Abductive Explanations for Boosted Regression Trees.
# Check V1.0: Ok
from pyxai import Builder, Explainer

node1_1 = Builder.DecisionNode(1, operator=Builder.GT, threshold=3000, left=1500, right=1750)
node1_2 = Builder.DecisionNode(1, operator=Builder.GT, threshold=2000, left=1000, right=node1_1)
node1_3 = Builder.DecisionNode(1, operator=Builder.GT, threshold=1000, left=0, right=node1_2)
tree1 = Builder.DecisionTree(5, node1_3)


node2_1 = Builder.DecisionNode(5, operator=Builder.EQ, threshold=1, left=100, right=250)
node2_2 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=-100, right=node2_1)
node2_3 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=node2_2, right=250)
tree2 = Builder.DecisionTree(5, node2_3)

node3_1 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=500, right=250)
node3_2 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=250, right=100)
node3_3 = Builder.DecisionNode(1, operator=Builder.GE, threshold=2000, left=0, right=node3_1)
node3_4 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=node3_3, right=node3_2)
tree3 = Builder.DecisionTree(5, node3_4)


BTs = Builder.BoostedTreesRegression([tree1, tree2, tree3])

instance = (2200, 0, 0, 1, 1) # 2200$, self employed (one hot encoded), married
print("instance:", instance)

explainer = Explainer.initialize(BTs, instance)

print("prediction:", explainer.predict(instance))
print("direct:", explainer.to_features(explainer.direct_reason()))

explainer.set_interval(1500, 2500)

tree_specific = explainer.tree_specific_reason()
print("tree specific:", explainer.to_features(tree_specific))
print("is tree : ", explainer.is_tree_specific_reason(tree_specific))

explainer.show()
#sufficient_reason = explainer.sufficient_reason()
#print("sufficient: ", sufficient_reason,  explainer.to_features(sufficient_reason))
#print("is implicant:", explainer.is_implicant(sufficient_reason))