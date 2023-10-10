# Check V1.0: Ok 

from pyxai import Builder, Explainer

node1 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=1)
node2 = Builder.DecisionNode(1, operator=Builder.GE, threshold=20, left=0, right=node1)
node3 = Builder.DecisionNode(1, operator=Builder.GE, threshold=30, left=node2, right=1)

tree1 = Builder.DecisionTree(2, node3)
tree2 = Builder.DecisionTree(2, Builder.LeafNode(1))

forest = Builder.RandomForest([tree1, tree2], n_classes=2)

alice = (18, 0)
explainer = Explainer.initialize(forest, instance=alice)
print("binary representation: ", explainer.binary_representation)
print("binary representation features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))
print("target_prediction:", explainer.target_prediction)

explainer = Explainer.initialize(forest, instance=alice, features_type={"numerical": ["f1"], "binary": ["f2"]})

contrastives = explainer.minimal_contrastive_reason(n=Explainer.ALL)
print("contrastives:", contrastives)
print("contrastives (to_features):", explainer.to_features(contrastives[0], contrastive=True))
print("contrastives (to_features):", explainer.to_features(contrastives[1], contrastive=True))

explainer.show()