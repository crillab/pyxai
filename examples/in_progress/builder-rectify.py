# Example taken from the paper quoted below.

# @misc{https://doi.org/10.48550/arxiv.2206.08758,
#   doi = {10.48550/ARXIV.2206.08758},
#   url = {https://arxiv.org/abs/2206.08758},
#   author = {Coste-Marquis, Sylvie and Marquis, Pierre},
#   keywords = {Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
#   title = {Rectifying Mono-Label Boolean Classifiers},
#   publisher = {arXiv},
#   year = {2022},  
#   copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
# }

from pyxai import Builder, Explainer

nodeT1_3 = Builder.DecisionNode(3, left=0, right=1)
nodeT1_2 = Builder.DecisionNode(2, left=1, right=0)
nodeT1_1 = Builder.DecisionNode(1, left=nodeT1_2, right=nodeT1_3)
tree = Builder.DecisionTree(3, nodeT1_1, force_features_equal_to_binaries=True)

print("type:", type(tree))
nodeT2_1 = Builder.DecisionNode(2, left=0, right=1)
positive_rectifying_tree = Builder.DecisionTree(3, nodeT2_1, force_features_equal_to_binaries=True)

nodeT3_3 = Builder.DecisionNode(3, left=0, right=1)
nodeT3_1 = Builder.DecisionNode(1, left=1, right=nodeT3_3)
negative_rectifying_tree = Builder.DecisionTree(3, nodeT3_1, force_features_equal_to_binaries=True)


explainer = Explainer.initialize(tree)
rectifying_tree = explainer.rectify(tree, positive_rectifying_tree, negative_rectifying_tree)

print("Original tree:", tree.raw_data_for_CPP())
print("Positive rectifying tree:", positive_rectifying_tree.raw_data_for_CPP())
print("Negative rectifying tree:", negative_rectifying_tree.raw_data_for_CPP())
print("Resulting tree:", rectifying_tree.raw_data_for_CPP())
