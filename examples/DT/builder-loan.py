# Example:
# Instances are described by three attributes: A (numerical), B_1 and B_2 (boolean).  
# The value of A gives the annual income of a customer (in k$)
# B_1 indicates whether the customer has no debts, and B_2 indicates that the customer has reimbursed his/her previous loan. 
# This is the classifier representing an approximation of the exact function. 
# The exact function, which we know, is: the loan is granted if and only if the client's annual income is at least $25k or if the client has no debts and has paid off his previous loan: 
# f(x) = 1 <=> (v1 >= 25) and ((v2 == 1) or (v3 == 1)). 
# @article{TODO}
# Check V1.0: Ok

from pyxai import Builder, Explainer

# Builder part

node_v3_1 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
node_v2_1 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=node_v3_1)

node_v3_2 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
node_v2_2 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=node_v3_2)

node_v3_3 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
node_v2_3 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=node_v3_3)

node_v1_1 = Builder.DecisionNode(1, operator=Builder.GE, threshold=10, left=node_v2_1, right=node_v2_2)
node_v1_2 = Builder.DecisionNode(1, operator=Builder.GE, threshold=20, left=node_v1_1, right=node_v2_3)
node_v1_3 = Builder.DecisionNode(1, operator=Builder.GE, threshold=30, left=node_v1_2, right=1)
node_v1_4 = Builder.DecisionNode(1, operator=Builder.GE, threshold=40, left=node_v1_3, right=1)

tree = Builder.DecisionTree(3, node_v1_4)

loan_types = {
    "numerical": ["f1"],
    "binary": ["f2", "f3"],
}

print("bob = (20, 1, 0):")
bob = (20, 1, 0)
explainer = Explainer.initialize(tree, instance=bob, features_type=loan_types)

print("binary representation: ", explainer.binary_representation)
print("target_prediction:", explainer.target_prediction)
print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))

minimals = explainer.minimal_sufficient_reason()
print("Minimal sufficient reasons:", minimals)
print("to_features:", explainer.to_features(minimals))

print("charles = (5, 0, 0):")
charles = (5, 0, 0)
explainer.set_instance(charles)

print("binary representation: ", explainer.binary_representation)
print("target_prediction:", explainer.target_prediction)
print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))

contrastives = explainer.contrastive_reason(n=Explainer.ALL)
for contrastive in contrastives:
    print("contrastive:", explainer.to_features(contrastive, contrastive=True))

explainer.show()