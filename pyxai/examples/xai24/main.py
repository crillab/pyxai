import random

from pyxai import Learning, Explainer, Tools
import cases
import user
import constants
import misc
import coverage
import time
import model
import matplotlib.pyplot as plt
random.seed(123)
Tools.set_verbose(0)


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
threshold = int(len(instances) * constants.classified_size)
classified_instances = instances[0:threshold]
test_instances = instances[threshold:-1]

if constants.user == constants.USER_BT:
    user = user.create_user_BT(AI)
if constants.user == constants.USER_LAMBDA:
    user = user.create_user_lambda(AI, classified_instances)


if constants.trace:
    print("nb positive rules", len(user.positive_rules))
    print("nb negative rules", len(user.negative_rules))

constants.statistics["n_positives"] = len(user.positive_rules)
constants.statistics["n_negatives"] = len(user.negative_rules)


# Statistics
cvg = coverage.Coverage(AI.model.get_theory([]), len(AI.explainer.binary_representation), 50, user)
accuracy_user = [user.accurary(test_instances)]
accuracy_AI = [misc.get_accuracy(AI.explainer.get_model(), test_instances)]
accuracy_AI_user = [misc.acuracy_wrt_user(user, AI.explainer, AI.explainer.get_model(), test_instances)]
coverages = [cvg.coverage()]
times = [0]
nodes_AI = [AI.model.n_nodes()]
all_cases = [None]
nb_binaries = [len(rule) for rule in user.positive_rules] + [len(rule) for rule in user.negative_rules]
print("c nb binaries at start:", nb_binaries)

nb_features = []
for rule in user.positive_rules:
    dict_features = {}
    tmp = user.explainer.to_features(rule,details=True, eliminate_redundant_features=False)
    for key in tmp.keys():
        if key not in dict_features:
            dict_features[key] = 1
    nb_features.append(len(dict_features.keys()))
for rule in user.negative_rules:
    dict_features = {}
    tmp = user.explainer.to_features(rule,details=True, eliminate_redundant_features=False)
    for key in tmp.keys():
        if key not in dict_features:
            dict_features[key] = 1
    nb_features.append(len(dict_features.keys()))

print("c nb features at start:", nb_features)


print("\n\n")

# Iterate on all classified instances


random.shuffle(classified_instances)
nb_instances = 100
for detailed_instance in classified_instances[0:nb_instances]:
    start_time = time.time()
    instance = detailed_instance['instance']
    AI.set_instance(instance)
    prediction_AI = AI.predict_instance(instance)
    prediction_user = user.predict_instance(AI.explainer.binary_representation)  # no they have the same representation
    rule_AI = AI.reason()
    # All cases
    print("user: ", prediction_user, "AI: ", prediction_AI)
    if prediction_user is None:  # cases (3) (4) (5)
         cas = cases.cases_3_4_5(AI.explainer, rule_AI, user)
         if cas == 3:
                constants.statistics["cases_3"] += 1
                all_cases.append(3)
         if cas == 4:
                constants.statistics["cases_4"] += 1
                all_cases.append(4)
         if cas == 5:
                constants.statistics["cases_5"] += 1
                all_cases.append(5)
    else:
        if prediction_AI != prediction_user:  # case (1)
            cases.case_1(AI.explainer, rule_AI, user)
            AI.set_instance(instance)
            if AI.explainer.target_prediction == prediction_AI:
                print("Aie aie aie")
            #assert(explainer_AI.target_prediction != prediction_AI)
            constants.statistics["cases_1"] += 1
            all_cases.append(1)

        if prediction_AI == prediction_user:  # case (2)
            cases.case_2(AI.explainer, rule_AI, user)
            constants.statistics["cases_2"] += 1
            all_cases.append(2)
    end_time = time.time()


    #  update statistics
    times.append(end_time - start_time)
    nodes_AI.append(AI.model.n_nodes())
    coverages.append(cvg.coverage())
    constants.statistics["n_positives"] = len(user.positive_rules)
    constants.statistics["n_negatives"] = len(user.negative_rules)
    accuracy_AI_user.append(misc.acuracy_wrt_user(user, AI.explainer, AI.explainer.get_model(), classified_instances))
    accuracy_user.append(user.accurary(classified_instances))
    accuracy_AI.append(misc.get_accuracy(AI.explainer.get_model(), classified_instances))


    if constants.trace:
        print("\n--\nc statistics", constants.statistics)
        print("\nc accuracy AI wrt user:", accuracy_AI_user)
        print("\nc accuracy user: ", accuracy_user)
        print("\nc accuracy AI:", accuracy_AI)
        print("\nc coverages:", coverages)
        print("\nc time:", times)
        print("\nc nodes:", nodes_AI)
        print("\nc cases:", all_cases)


nb_binaries = [len(rule) for rule in user.positive_rules] + [len(rule) for rule in user.negative_rules]
print("c nb binaries at end:", nb_binaries)

nb_features = []
for rule in user.positive_rules:
    dict_features = {}
    tmp = AI.explainer.to_features(rule,details=True, eliminate_redundant_features=False)
    for key in tmp.keys():
        if key not in dict_features:
            dict_features[key] = 1
    nb_features.append(len(dict_features.keys()))
for rule in user.negative_rules:
    dict_features = {}
    tmp = AI.explainer.to_features(rule,details=True, eliminate_redundant_features=False)
    for key in tmp.keys():
        if key not in dict_features:
            dict_features[key] = 1
    nb_features.append(len(dict_features.keys()))

print("c nb features at end:", nb_features)



