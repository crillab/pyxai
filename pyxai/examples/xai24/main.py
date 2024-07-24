import random

from pyxai import Learning, Explainer, Tools
import cases
import user as us
import constants
import misc
import coverage
import time
import model
import matplotlib.pyplot as plt
import sys
random.seed(123)
Tools.set_verbose(1)
import time

sys.setrecursionlimit(100000)

global_time = time.time()

if Tools.Options.n is not None:
    constants.N = int(Tools.Options.n)
print("N = ", constants.N)
assert(constants.model == Learning.RF or constants.model == Learning.DT)

# create AI
print("Create AI: ", "DT" if constants.model == Learning.DT else "RF")
learner_AI = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
AI = model.Model(learner_AI)
AI.explainer = Explainer.initialize(AI.model, features_type=Tools.Options.types)
instances = learner_AI.get_instances(AI.model, indexes=Learning.TEST, details=True)

# Extract test instances and classified instances
threshold_1 = int(len(instances) * constants.user_size)
threshold_2 = int(len(instances) * constants.interaction_size)
threshold_3 = int(len(instances) * constants.test_size)

#print("Total test instances:", len(instances))

user_instances = instances[0:threshold_1]
interaction_instances = instances[threshold_1:threshold_1+threshold_2]
test_instances = instances[threshold_1+threshold_2:-1]
#test_instances = user_instances
print("Total instances:", learner_AI.n_instances)
print("Total training instances:", learner_AI.n_instances -  len(instances))
print("Total user_instances:", len(user_instances))
print("Total interaction_instances:", len(interaction_instances))
print("Total test_instances:", len(test_instances))


if constants.user == constants.USER_BT:
    user = us.create_user_BT(AI)
if constants.user == constants.USER_LAMBDA:
    user = us.create_user_lambda_forest(AI, user_instances)


if constants.trace:
    print("nb positive rules", len(user.positive_rules))
    print("nb negative rules", len(user.negative_rules))

constants.statistics["n_positives"] = len(user.positive_rules)
constants.statistics["n_negatives"] = len(user.negative_rules)

constants.statistics["n_initial_positives"] = len(user.positive_rules)
constants.statistics["n_initial_negatives"] = len(user.negative_rules)

print("positive rules:", user.positive_rules)
print("negative rules:", user.negative_rules)


#Rend debile l'IA:
print("nTrees IA: ", len(AI.model.forest))
print("accuracy IA: ", misc.get_accuracy(AI.explainer.get_model(), test_instances))

AI.model.forest = AI.model.forest[0:1]
AI.explainer = Explainer.initialize(AI.model, features_type=Tools.Options.types)
AI.set_instance(test_instances[0]["instance"])
print("new nTrees IA: ", len(AI.model.forest))
print("new accuracy IA: ", misc.get_accuracy(AI.explainer.get_model(), test_instances))
#exit(0)
# Statistics
cvg = coverage.Coverage(AI.model.get_theory([]), len(AI.explainer.binary_representation), 1, user)
accuracy_user = [user.accurary(test_instances)]
accuracy_AI = [misc.get_accuracy(AI.explainer.get_model(), test_instances)]
accuracy_AI_user = [misc.acuracy_wrt_user(user, AI.explainer, AI.explainer.get_model(), test_instances)]
accuracy_AI_interaction = [misc.get_accuracy(AI.explainer.get_model(), interaction_instances)]
accuracy_user_interaction = [user.accurary(interaction_instances)]

coverages = [cvg.coverage()]
times = [0]
nodes_AI = [AI.model.n_nodes()]
all_cases = [None]


misc.print_features(user)
print("\n\n")

# Iterate on all classified instances


random.shuffle(interaction_instances)
nb_instances = 100
for detailed_instance in interaction_instances[0:nb_instances]:
    if time.time() - global_time > constants.max_time:
        print("No more time")
        sys.exit(1)
    start_time = time.time()
    instance = detailed_instance['instance']
    AI.set_instance(instance)
    #print("instance:", instance)
    prediction_AI = AI.predict_instance(instance)
    prediction_user = user.predict_instance(AI.explainer.binary_representation)  # no they have the same representation
    rule_AI = AI.reason()
    #print("rule AI", rule_AI)
    #print("positive", user.positive_rules)
    #print("negative", user.negative_rules)
    # All cases
    print("user: ", prediction_user, "AI: ", prediction_AI)
    if prediction_user is None:  # cases (3) (4)
         cas = cases.cases_3_4(AI.explainer, rule_AI, user)
         if cas == 3:
                constants.statistics["cases_3"] += 1
                all_cases.append(3)
         elif cas == 4:
                constants.statistics["cases_4"] += 1
                all_cases.append(4)
         #if cas == 5:
         #       constants.statistics["cases_5"] += 1
         #       all_cases.append(5)
    else:
        if prediction_AI != prediction_user:  # case (1)
            cases.case_1(AI.explainer, rule_AI, user)
            AI.set_instance(instance)
            if AI.explainer.target_prediction == prediction_AI:
                print("Aie aie aie")
            #assert(explainer_AI.target_prediction != prediction_AI)
            constants.statistics["cases_1"] += 1
            all_cases.append(1)

        if prediction_AI == prediction_user:  # case (2)(4)
            cas = cases.case_2(AI.explainer, rule_AI, user)
            if cas == 2:
                constants.statistics["cases_2"] += 1
                all_cases.append(2)
            elif cas == 4:
                constants.statistics["cases_4"] += 1
                all_cases.append(4)
            
    """
    for rule1 in user.positive_rules:
        for rule2 in user.negative_rules:
            assert(us.conflict(AI.explainer, rule1, rule2) is False)
    for rule1 in user.negative_rules:
        for rule2 in user.positive_rules:
            assert (us.conflict(AI.explainer, rule1, rule2) is False)
    """
    end_time = time.time()


    #  update statistics
    times.append(end_time - start_time)
    nodes_AI.append(AI.model.n_nodes())
    coverages.append(cvg.coverage())
    constants.statistics["n_positives"] = len(user.positive_rules)
    constants.statistics["n_negatives"] = len(user.negative_rules)
    accuracy_AI_user.append(misc.acuracy_wrt_user(user, AI.explainer, AI.explainer.get_model(), test_instances))
    accuracy_user.append(user.accurary(test_instances))
    accuracy_AI.append(misc.get_accuracy(AI.explainer.get_model(), test_instances))
    accuracy_AI_interaction.append(misc.get_accuracy(AI.explainer.get_model(), interaction_instances))
    accuracy_user_interaction.append(user.accurary(interaction_instances))
    #exit(0)
    if constants.trace:
        print("\n--\nc statistics", constants.statistics)
        print("\nc accuracy AI wrt user:", accuracy_AI_user)
        print("\nc accuracy user: ", accuracy_user)
        print("\nc accuracy AI:", accuracy_AI)
        print("\nc accuracy_AI_interaction:", accuracy_AI_interaction)
        print("\nc accuracy_user_interaction:", accuracy_user_interaction)
        print("\nc coverages:", coverages)
        print("\nc time:", times)
        print("\nc nodes:", nodes_AI)
        print("\nc cases:", all_cases)
        

misc.print_features(user)


