
from pyxai import Builder, Explainer

tree_1_b = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=1, right=0)
tree_1_a = Builder.DecisionNode(1, operator=Builder.EQ, threshold=1, left=0, right=tree_1_b)
tree_1 = Builder.DecisionTree(4, tree_1_a, force_features_equal_to_binaries=True)

tree_2_d = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=1, right=0)
tree_2_c = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=tree_2_d, right=0)
tree_2 = Builder.DecisionTree(4, tree_2_c, force_features_equal_to_binaries=True)

tree_3_d = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_3_c = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=tree_3_d, right=0)
tree_3_a = Builder.DecisionNode(1, operator=Builder.EQ, threshold=1, left=0, right=tree_3_c)
tree_3 = Builder.DecisionTree(4, tree_3_a, force_features_equal_to_binaries=True)

tree_4_d = Builder.DecisionNode(4, operator=Builder.EQ, threshold=1, left=0, right=1)
tree_4_c = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=tree_4_d)
tree_4_b = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=tree_4_c, right=0)
tree_4 = Builder.DecisionTree(4, tree_4_b, force_features_equal_to_binaries=True)

unit_tree_1 = Builder.DecisionTree(4, Builder.LeafNode(1), force_features_equal_to_binaries=True)
unit_tree_2 = Builder.DecisionTree(4, Builder.LeafNode(1), force_features_equal_to_binaries=True)
unit_tree_3 = Builder.DecisionTree(4, Builder.LeafNode(1), force_features_equal_to_binaries=True)

forest = Builder.RandomForest([tree_1, tree_2, tree_3, tree_4, unit_tree_1, unit_tree_2, unit_tree_3], n_classes=2, feature_names=["a", "b", "c", "d"])


# paper part
print("x = (1, 0, 0, 0):")
instance = (1, 0, 0, 0)

explainer = Explainer.initialize(forest, features_type={"binary": ["a", "b", "c", "d"],})
explainer.set_instance(instance=instance)
print("intance:", instance)
print("binary representation: ", explainer.binary_representation)
print("target_prediction:", explainer.target_prediction)
print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))

reference_instances = {0:[(0,0,0,0),(0,0,1,0),(0,1,1,0)],
        1:[(0,0,1,1),(1,0,1,0),(1,1,0,0),(1,1,0,1)]}

minimal = explainer.minimal_sufficient_reason()
print("minimal:", explainer.to_features(minimal))
anchored_reason = explainer.anchored_reason(n_anchors=2, reference_instances=reference_instances, check=True)
print("anchored_reason:", explainer.to_features(anchored_reason))

# check part
for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            for d in [0, 1]:
                instance = (a, b, c, d)
                explainer.set_instance(instance=instance)
                print("-------------------------")
                print("intance:", instance)
                print("binary representation: ", explainer.binary_representation)
                print("target_prediction:", explainer.target_prediction)
                print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))
                if explainer.target_prediction == 0:
                    reference_instances = {0:[instance], 1:[[0 if l==1 else 1 for l in instance]]}
                else:
                    reference_instances = {0:[[0 if l==1 else 1 for l in instance]], 1:[instance]}
                print("reference_instances:", reference_instances)

                n_anchors = 1
                go_next = True
                while(go_next is True):
                    anchored_reason = explainer.anchored_reason(n_anchors=n_anchors, reference_instances=reference_instances, check=True)
                    print(str(n_anchors)+"-anchored_reason:"+str(anchored_reason))
                    if anchored_reason is not None:
                        go_next = True
                        n_anchors += 1
                    else:
                        break