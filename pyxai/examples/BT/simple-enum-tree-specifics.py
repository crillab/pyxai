from pyxai import Learning, Explaining, Tools
import numpy
# Check V1.0: Ok
# Check V2.0: Ok

# Machine learning part
learner = Learning.Xgboost(Tools.Options.dataset, problem_type='classification', instances_type='tabular', labels_type='classes')
model = learner.evaluate(splitting_method=Learning.HOLD_OUT, model_type=Learning.BT, model_parameters={}, splitting_parameters={})
instances = learner.get_instances(model=model, n=10, is_correct=True)

# Explanation part


for instance, p in instances:
    explainer1 = Explaining.initialize(model, instance, features_type=Tools.Options.types) 
    #explainer2 = Explaining.initialize(model, instance) 
    tree_specific_reason1 = explainer1.minimal_tree_specific_reasons(n=1)
    #tree_specific_reason2 = explainer2.minimal_tree_specific_reasons(n=1)

    #print(tree_specific_reason1)
    #print(tree_specific_reason2)
    # print("is a tree specific", explainer1.is_tree_specific_reason(tree_specific_reason1))
    #print("is a tree specific", explainer2.is_tree_specific_reason(tree_specific_reason2))
    print("is_implicant", explainer1.is_implicant(tree_specific_reason1))
    #print("is_implicant", explainer2.is_implicant(tree_specific_reason2))

#instances = learner.get_instances(n=100)

#print(instances)
#for inst, p in instances:
#    explainer.set_instance(inst)
#    direct_reason = explainer.direct_reason()
#
#    tree_specific_reason = explainer.tree_specific_reason(n_iterations=100)
#    print("is a tree specific", explainer.is_tree_specific_reason(tree_specific_reason))

#minimal_tree_specific_reason = explainer.minimal_tree_specific_reason(time_limit=20)
#print("\nlen minimal tree_specific: ", len(minimal_tree_specific_reason))
#print("is a tree specific", explainer.is_tree_specific_reason(minimal_tree_specific_reason))
#if explainer.elapsed_time == Explainer.TIMEOUT: print("Not minimal, this is an approximation")
# s = explainer.sufficient_reason()

