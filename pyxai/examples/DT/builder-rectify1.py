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

#Let us suppose that the predictor f furnished by the bank labels an instance positive when it corresponds to a customer who
# has high incomes (f1) but has not reimbursed a previous loan (f2),
# or (which looks more risky) a customer who has low incomes (f1) and has some debts (f3).

from pyxai import Builder, Explainer

nodeT1_3 = Builder.DecisionNode(3, left=0, right=1)
nodeT1_2 = Builder.DecisionNode(2, left=1, right=0)
nodeT1_1 = Builder.DecisionNode(1, left=nodeT1_2, right=nodeT1_3)
model = Builder.DecisionTree(3, nodeT1_1, force_features_equal_to_binaries=True)

loan_types = {
    "binary": ["f1", "f2", "f3"],
}

explainer = Explainer.initialize(model, features_type=loan_types)

print("Original tree:", explainer.get_model().raw_data_for_CPP())

#Alice’s expertise can be represented by the formula T = ((x1 ∧ not x3) ⇒ y) ∧ (not x2 ⇒ not y) encoding her two decision rules
explainer.rectify(conditions=(1, -3), label=1)  #(x1 ∧ not x3) ⇒ y
explainer.rectify(conditions=(-2, ), label=0)  #not x2 ⇒ not y

rectified_model = explainer.get_model().raw_data_for_CPP()
print("Rectified_model:", rectified_model)

assert (0, (1, 0, (2, 0, 1))) == rectified_model, "The rectified model is not good."



