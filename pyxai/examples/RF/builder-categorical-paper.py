# Example:
# Instances are described by three attributes: A (numerical), B_1 and B_2 (boolean).  
# The value of A gives the annual income of a customer (in k$)
# B_1 indicates whether the customer has no debts, and B_2 indicates that the customer has reimbursed his/her previous loan. 
# This is the classifier representing an approximation of the exact function. 
# The exact function, which we know, is: the loan is granted if and only if the client's annual income is at least $25k or if the client has no debts and has paid off his previous loan: f(x) = 1 <=> (v1 >= 25) and ((v2 == 1) or (v3 == 1)). 
# @article{TODO}
# Check V1.0: Ok
from pyxai import Builder, Explainer, Learning

# Builder part

node_t1_v1_1 = Builder.DecisionNode(1, operator=Builder.GE, threshold=10, left=0, right=0)
node_t1_v1_2 = Builder.DecisionNode(1, operator=Builder.GE, threshold=20, left=node_t1_v1_1, right=0)
node_t1_v1_3 = Builder.DecisionNode(1, operator=Builder.GE, threshold=30, left=node_t1_v1_2, right=1)
node_t1_v1_4 = Builder.DecisionNode(1, operator=Builder.GE, threshold=40, left=node_t1_v1_3, right=1)
tree_1 = Builder.DecisionTree(3, node_t1_v1_4)

node_t2_v3_1 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=1)
node_t2_v3_2 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=node_t2_v3_1)
#node_t2_v3_3 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=3, left=node_t2_v3_2, right=1)
#node_t2_v2 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=node_t2_v3_3)

tree_2 = Builder.DecisionTree(3, node_t2_v3_2)

tree_3 = Builder.DecisionTree(3, Builder.LeafNode(1))

forest = Builder.RandomForest([tree_1, tree_2, tree_3], n_classes=2)
#forest.add_numerical_feature(1)
#forest.add_categorical_feature(3)


print("numerical_features:", forest.numerical_features)

print("bob = (20, 1, 0):")
bob = (20, 1, 0)
explainer = Explainer.initialize(forest, instance=bob, features_type={"numerical":["f1"], "binary":Learning.DEFAULT})

print("binary representation: ", explainer.binary_representation)
print("target_prediction:", explainer.target_prediction)
print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))

minimals = explainer.minimal_sufficient_reason()
print("Minimal sufficient reasons:", minimals)
print("to_features:", explainer.to_features(minimals, eliminate_redundant_features=True))

print()
print("charles = (5, 0, 0):")
charles = (5, 0, 0)
explainer.set_instance(charles)

print("binary representation: ", explainer.binary_representation)
print("target_prediction:", explainer.target_prediction)
print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))

contrastive = explainer.minimal_contrastive_reason(n=1)
print("contrastive:", contrastive)

print("contrastive (to_features):", explainer.to_features(contrastive, contrastive=True))
print("is contrastive:", explainer.is_contrastive_reason(contrastive))

explainer.show()