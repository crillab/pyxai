
from pyxai import Builder, Explainer

node_f1 = Builder.DecisionNode(1, operator=Builder.EQ, threshold=1, left=0, right=1)
node_f2 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=1, right=0)
node_f3 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=1, right=0)

node_f1_1 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=1, right=node_f1)

node_f2_1 = Builder.DecisionNode(1, operator=Builder.EQ, threshold=1, left=0, right=node_f2)

node_f3_1 = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=node_f2_1, right=node_f3)

node_r = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=node_f1_1, right=node_f3_1)


tree = Builder.DecisionTree(4, node_r, force_features_equal_to_binaries=True, feature_names=["a", "b", "c", "d"])

print("x = (1, 0, 0, 0):")
x = (1, 0, 0, 0)
explainer = Explainer.initialize(tree, instance=x, features_type={"binary": ["a", "b", "c", "d"],})

print("binary representation: ", explainer.binary_representation)
print("target_prediction:", explainer.target_prediction)
print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))

print()
print("compute minimals:")
minimals = explainer.minimal_sufficient_reason(n=10)
for minimal in minimals:
    print("Minimal sufficient reasons:", minimal)
    print("to_features:", explainer.to_features(minimal))

reference_instances = {0:[(0,0,0,0),(0,0,1,0),(0,1,1,0)],
                       1:[(0,0,1,1),(1,0,1,0),(1,1,0,0),(1,1,0,1)]}

explainer.anchored_reason(n_anchors=2, reference_instances=reference_instances)
