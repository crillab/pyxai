# Check V2.0: Not OK (But Encore must be compiled) 'ExplainerDT' object has no attribute '_most_anchored_reason_anchored_reason'

from pyxai import Builder, Learning, Explaining

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

learner_information = Learning.LearnerInformation(None, problem_type='classification')
forest = Builder.RandomForest([tree_1, tree_2, tree_3], n_classes=2, learner_information=learner_information)

explainer = Explaining.initialize(forest)
explainer.set_instance(instance=(0,1,1))
reference_instances = {0:[(0,0,0)],
        1:[(1,1,1)]}

explainer.set_reference_instances(reference_instances)
anchored_reason = explainer.most_anchored_reason(check=True)

print("anchored_reason:", anchored_reason)
print("anchored_reason:", explainer.to_features(anchored_reason))

# check part (Ok V2)
# for a in [0, 1]:
#     for b in [0, 1]:
#         for c in [0, 1]:
                
#                 instance = (a, b, c)
#                 #if instance == (0,1,1):
#                 explainer.set_instance(instance=instance)
#                 print("-------------------------")
#                 print("intance:", instance)
#                 print("binary representation: ", explainer.binary_representation)
#                 print("target_prediction:", explainer.target_prediction)
#                 print("to_features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))
#                 if explainer.target_prediction == 0:
#                     reference_instances = {0:[instance], 1:[[0 if l==1 else 1 for l in instance]]}
#                 else:
#                     reference_instances = {0:[[0 if l==1 else 1 for l in instance]], 1:[instance]}
#                 print("reference_instances:", reference_instances)

#                 n_anchors = 1
#                 go_next = True
#                 while(go_next is True):
#                     explainer.set_reference_instances(reference_instances)
#                     anchored_reason = explainer.most_anchored_reason(check=True)
#                     print(str(n_anchors)+"-anchored_reason:"+str(anchored_reason))
#                     if anchored_reason is not None:
#                         go_next = True
#                         n_anchors += 1
#                     else:
#                         break