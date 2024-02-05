
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
x = (1, 1, 1, 1)

explainer = Explainer.initialize(tree, features_type={"binary": ["a", "b", "c", "d"],})
                
for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            for d in [0, 1]:
                instance = (a, b, c, d)
                explainer.set_instance(instance=instance)
                print("intance:", instance)
                print("binary representation: ", explainer.binary_representation)
                print("target_prediction:", explainer.target_prediction)
                print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))

                reference_instances = {0:[(0,0,0,0),(0,0,1,0),(0,1,1,0)],
                       1:[(0,0,1,1),(1,0,1,0),(1,1,0,0),(1,1,0,1)]}

                n_anchors = 0
                go_next = True
                while(go_next is True):
                    anchored_reason = explainer.anchored_reason(n_anchors=n_anchors, reference_instances=reference_instances, check=True)
                    print(str(n_anchors)+"-anchored_reason:"+str(anchored_reason))
                    if anchored_reason is not None:
                        go_next = True
                        n_anchors += 1
                    else:
                        break

for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            for d in [0, 1]:
                instance = (a, b, c, d)
                explainer.set_instance(instance=instance)
                print("intance:", instance)
                print("binary representation: ", explainer.binary_representation)
                print("target_prediction:", explainer.target_prediction)
                print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))

                reference_instances = {0:[(0,0,0,0),(0,0,1,0),(0,1,1,0),(1,0,0,1)],
                       1:[(0,0,1,1),(1,0,1,0),(1,1,0,0),(1,1,0,1)]}

                n_anchors = 0
                go_next = True
                while(go_next is True):
                    anchored_reason = explainer.anchored_reason(n_anchors=n_anchors, reference_instances=reference_instances, check=True)
                    print(str(n_anchors)+"-anchored_reason:"+str(anchored_reason))
                    if anchored_reason is not None:
                        go_next = True
                        n_anchors += 1
                    else:
                        break

#print("to_features:", explainer.to_features(anchored_reason))

#reference_instances = {0:[(0,0,0,0),(0,0,1,0),(0,1,1,0),(1,0,0,1)],
#                       1:[(0,0,1,1),(1,0,1,0),(1,1,0,0),(1,1,0,1)]}

#anchored_reason = explainer.anchored_reason(n_anchors=1, reference_instances=reference_instances, check=True)
#print("1-anchored_reason:", anchored_reason)

#anchored_reason = explainer.anchored_reason(n_anchors=2, reference_instances=reference_instances, check=True)
#print("2-anchored_reason:", anchored_reason)
