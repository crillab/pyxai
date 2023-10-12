from pyxai import Builder, Learning, Explainer, Tools
import math


Tools.set_verbose(0)

import unittest
class TestToFeatures(unittest.TestCase):

    def test_to_features(self):
        node_t1_v1_1 = Builder.DecisionNode(1, operator=Builder.GE, threshold=0, left=0, right=0)
        node_t1_v1_1_1 = Builder.DecisionNode(1, operator=Builder.GE, threshold=10, left=node_t1_v1_1, right=0)
        node_t1_v1_2 = Builder.DecisionNode(1, operator=Builder.GE, threshold=20, left=node_t1_v1_1_1, right=0)
        node_t1_v1_3 = Builder.DecisionNode(1, operator=Builder.GT, threshold=30, left=node_t1_v1_2, right=1)
        node_t1_v1_4 = Builder.DecisionNode(1, operator=Builder.GE, threshold=40, left=node_t1_v1_3, right=1)
        tree1 = Builder.DecisionTree(1, node_t1_v1_4)
        
        tree2 = Builder.DecisionTree(1, Builder.LeafNode(1))

        forest = Builder.RandomForest([tree1, tree2], n_classes=2)

        alice = (18,)
        #print("alice:", alice)
        explainer = Explainer.initialize(forest, instance=alice)
        #print("binary representation: ", explainer.binary_representation)
        #print("binary representation features:", explainer.to_features(explainer.binary_representation, eliminate_redundant_features=False))
        #print("target_prediction:", explainer.target_prediction)

        explainer = Explainer.initialize(forest, instance=alice, features_type={"numerical": ["f1"]})

        test_1 = explainer.binary_representation
        #print("test_1:", test_1)
        #print("test_1:", explainer.to_features(test_1))
        self.assertEqual(explainer.to_features(test_1),('f1 in [10, 20[',)) 
        
        test_2 = explainer.minimal_contrastive_reason(n=1)
        
        #print("test_2:", test_2)
        #print("test_2:", explainer.to_features(test_2, contrastive=True))
        self.assertEqual(explainer.to_features(test_2, contrastive=True),('f1 <= 30',)) 

        test_3 = explainer.minimal_sufficient_reason()
        #print("test_3:", test_3)
        #print("test_3:", explainer.to_features(test_3))
        self.assertEqual(explainer.to_features(test_3),('f1 < 20',)) 
        
if __name__ == '__main__':
    print("Tests: " + TestToFeatures.__name__ + ":")
    unittest.main()