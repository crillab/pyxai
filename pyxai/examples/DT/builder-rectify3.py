
from pyxai import Builder, Explainer

# Builder part

node_L_1 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
node_L_2 = Builder.DecisionNode(1, operator=Builder.GT, threshold=20, left=0, right=node_L_1)

node_R_1 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
node_R_2 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=node_R_1, right=1)

root = Builder.DecisionNode(1, operator=Builder.GT, threshold=30, left=node_L_2, right=node_R_2)
tree = Builder.DecisionTree(3, root, feature_names=["I", "PP", "R"])

print("base:", tree.raw_data_for_CPP())
loan_types = {
    "numerical": ["I"],
    "binary": ["PP", "R"],
}

explainer = Explainer.initialize(tree, instance=(25, 1, 1), features_type=loan_types)

print("binary representation: ", explainer.binary_representation)
print("target_prediction:", explainer.target_prediction)
print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))

#For him/her, the following classification rule must be obeyed:
#whenever the annual income of the client is lower than 30,
#the demand should be rejected
rectified_model = explainer.rectify(conditions=(-1, ), label=0) 

assert (0, (1, 0, (4, (3, 0, 1), 1))) == rectified_model.raw_data_for_CPP(), "The rectified model is not good."
