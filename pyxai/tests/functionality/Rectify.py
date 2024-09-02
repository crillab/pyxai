from pyxai import Builder, Learning, Explainer, Tools
import math


Tools.set_verbose(0)


import unittest
class TestRectify(unittest.TestCase):
    
    #@unittest.skip("reason for skipping")
    def test_rectify_5(self):
        learner = Learning.Scikitlearn("tests/compas.csv", learner_type=Learning.CLASSIFICATION)
        model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, n_estimators=1)

        dict_information = learner.get_instances(model, n=1, indexes=Learning.TEST, correct=False, details=True)
        
        all_dict_information = learner.get_instances(model, indexes=Learning.ALL, details=True)

        instance = dict_information["instance"]
        label = dict_information["label"]
        prediction = dict_information["prediction"]
        
        compas_types = {
            "numerical": ["Number_of_Priors"],
            "binary": ["Misdemeanor", "score_factor", "Female"],
            "categorical": {"Origin*": ["African_American", "Asian", "Hispanic", "Native_American", "Other"],
                            "Age*": ["Above_FourtyFive", "Below_TwentyFive"]}
        }


        explainer = Explainer.initialize(model, instance=instance, features_type=compas_types)
        reason = explainer.majoritary_reason(n=1)
        #print("reason:", reason)
        #print("reason:", explainer.to_features(reason))
        reason = tuple([(2, Builder.GT, 0.5, True)] + [r for r in reason][1:])
        #print("reason:", reason)
        
        model = explainer.rectify(conditions=reason, label=1, tests=True)
        reason = (-2, -3, -7, 9)
        self.assertEqual(model.predict_instance(instance), 1)
        
        reason = set(reason)
        
        for instance_dict in all_dict_information:
            instance = instance_dict["instance"]
            old_prediction = instance_dict["prediction"]
            binary_representation = set(explainer._to_binary_representation(instance))
            result = binary_representation.intersection(reason)
            if len(result) == len(reason):
                self.assertEqual(model.predict_instance(instance), 1)
            else:
                self.assertEqual(model.predict_instance(instance), old_prediction)

    #@unittest.skip("reason for skipping")
    def test_rectify_b(self):
        node_v3_1 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
        node_v2_1 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=node_v3_1)
        
        node_v1_1 = Builder.DecisionNode(1, operator=Builder.GE, threshold=40, left=node_v2_1, right=0)
        node_v1_2 = Builder.DecisionNode(1, operator=Builder.GE, threshold=30, left=node_v1_1, right=0)
        node_v1_3 = Builder.DecisionNode(1, operator=Builder.GE, threshold=20, left=node_v1_2, right=0)
        node_v1_4 = Builder.DecisionNode(1, operator=Builder.GE, threshold=10, left=node_v1_3, right=1)

        model = Builder.DecisionTree(3, node_v1_4)

        loan_types = {
            "numerical": ["f1"],
            "binary": ["f2", "f3"],
        }

        bob = (20, 1, 0)

        explainer = Explainer.initialize(model, instance=bob, features_type=loan_types)
        explainer.rectify(conditions=((1, Builder.GE, 5, False),-5), label=1, tests=True) 
        rectified_model = explainer.get_model().raw_data_for_CPP()
        
    #@unittest.skip("reason for skipping")
    def test_rectify_a(self):
        node_v3_1 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
        node_v2_1 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=node_v3_1)
        
        node_v1_1 = Builder.DecisionNode(1, operator=Builder.GE, threshold=40, left=node_v2_1, right=0)
        node_v1_2 = Builder.DecisionNode(1, operator=Builder.GE, threshold=30, left=node_v1_1, right=0)
        node_v1_3 = Builder.DecisionNode(1, operator=Builder.GE, threshold=20, left=node_v1_2, right=0)
        node_v1_4 = Builder.DecisionNode(1, operator=Builder.GE, threshold=10, left=node_v1_3, right=1)

        tree = Builder.DecisionTree(3, node_v1_4)

        model = Builder.RandomForest([tree])
        loan_types = {
            "numerical": ["f1"],
            "binary": ["f2", "f3"],
        }

        bob = (20, 1, 0)
        explainer = Explainer.initialize(model, instance=bob, features_type=loan_types)

        
        minimal = explainer.minimal_sufficient_reason()
        #print("minimal:", minimal)
        #print("minimal:", explainer.to_features(minimal))
        

        
        explainer.rectify(conditions=minimal, label=1, tests=True) 

    #@unittest.skip("reason for skipping")
    def test_rectify_1(self):
        nodeT1_3 = Builder.DecisionNode(3, left=0, right=1)
        nodeT1_2 = Builder.DecisionNode(2, left=1, right=0)
        nodeT1_1 = Builder.DecisionNode(1, left=nodeT1_2, right=nodeT1_3)
        model = Builder.DecisionTree(3, nodeT1_1, force_features_equal_to_binaries=True)

        loan_types = {
            "binary": ["f1", "f2", "f3"],
        }

        explainer = Explainer.initialize(model, features_type=loan_types)

        #Alice’s expertise can be represented by the formula T = ((x1 ∧ not x3) ⇒ y) ∧ (not x2 ⇒ not y) encoding her two decision rules
        explainer.rectify(conditions=(1, -3), label=1, tests=True)  #(x1 ∧ not x3) ⇒ y
        explainer.rectify(conditions=(-2, ), label=0, tests=True)  #not x2 ⇒ not y

        rectified_model = explainer.get_model().raw_data_for_CPP()
        
        self.assertEqual(rectified_model, (0, (1, 0, (2, 0, 1))))
    
    #@unittest.skip("reason for skipping")
    def test_rectify_2(self):
        
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

        bob = (20, 1, 0)
        explainer = Explainer.initialize(tree, instance=bob, features_type=loan_types)

        
        minimal = explainer.minimal_sufficient_reason()

        
        explainer.rectify(conditions=minimal, label=1, tests=True) 
        rectified_model = explainer.get_model().raw_data_for_CPP()
        self.assertEqual(rectified_model, (0, (1, (2, (5, (6, 1, 0), 1), 1), 1)))

    #@unittest.skip("reason for skipping")
    def test_rectify_4(self):
        
        node_L_1 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
        node_L_2 = Builder.DecisionNode(1, operator=Builder.GT, threshold=20, left=0, right=node_L_1)

        node_R_1 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
        node_R_2 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=node_R_1, right=1)

        root = Builder.DecisionNode(1, operator=Builder.GT, threshold=30, left=node_L_2, right=node_R_2)
        tree = Builder.DecisionTree(3, root, feature_names=["I", "PP", "R"])

        loan_types = {
            "numerical": ["I"],
            "binary": ["PP", "R"],
        }

        bob = (25, 1, 1)
        explainer = Explainer.initialize(tree, instance=bob, features_type=loan_types)

        
        #For him/her, the following classification rule must be obeyed:
        #whenever the annual income of the client is lower than 30,
        #the demand should be rejected
        rectified_model = explainer.rectify(conditions=(-1, ), label=0, tests=True) 

        self.assertEqual(rectified_model.raw_data_for_CPP(), (0, (1, 0, (4, (3, 0, 1), 1))))
    
    #@unittest.skip("reason for skipping")
    def test_rectify_3(self):
        learner = Learning.Scikitlearn("tests/compas.csv", learner_type=Learning.CLASSIFICATION)
        model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)

        dict_information = learner.get_instances(model, n=1, indexes=Learning.TEST, correct=False, details=True)
        
        all_dict_information = learner.get_instances(model, indexes=Learning.ALL, details=True)

        instance = dict_information["instance"]
        label = dict_information["label"]
        prediction = dict_information["prediction"]
        
      
        compas_types = {
            "numerical": ["Number_of_Priors"],
            "binary": ["Misdemeanor", "score_factor", "Female"],
            "categorical": {"Origin*": ["African_American", "Asian", "Hispanic", "Native_American", "Other"],
                            "Age*": ["Above_FourtyFive", "Below_TwentyFive"]}
        }


        explainer = Explainer.initialize(model, instance=instance, features_type=compas_types)
        minimal_reason = explainer.minimal_sufficient_reason(n=1)
        model = explainer.rectify(conditions=minimal_reason, label=1, tests=True)
        
        self.assertEqual(model.predict_instance(instance), 1)
        
        reason = set(minimal_reason)
        
        for instance_dict in all_dict_information:
            instance = instance_dict["instance"]
            old_prediction = instance_dict["prediction"]
            binary_representation = set(explainer._to_binary_representation(instance))
            result = binary_representation.intersection(reason)
            if len(result) == len(reason):
                self.assertEqual(model.predict_instance(instance), 1)
            else:
                self.assertEqual(model.predict_instance(instance), old_prediction)
    
    
        
if __name__ == '__main__':
    print("Tests: " + TestRectify.__name__ + ":")
    unittest.main()