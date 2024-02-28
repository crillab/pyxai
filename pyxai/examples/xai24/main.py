from pyxai import Learning, Explainer, Tools
import cases
import user
import constants
import misc
import coverage

#Tools.set_verbose(0)
# Create the user agent
learner_user = Learning.Xgboost(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model_user = learner_user.evaluate(method=Learning.HOLD_OUT, output=Learning.BT, test_size=1 - constants.training_size, seed=123)
instances = learner_user.get_instances(model_user, indexes=Learning.TEST, details=True)

# Change weights of BT
if constants.debug:
    print("Accuracy before", misc.get_accuracy(model_user, test_set=instances[0:200]))
misc.change_weights(model_user)
if constants.debug:
    print("Accuracy after ", misc.get_accuracy(model_user, instances[0:200]))


# Extract test instances and classified instances
threshold = int(len(instances) * constants.classified_size)
classified_instances = instances[0:threshold]
test_instances = instances[threshold:-1]
positive_instances, negative_instances, unclassified_instances = misc.partition_instances(model_user, classified_instances)

if constants.trace:
    print("nb positives:", len(positive_instances))
    print("nb negatives:", len(negative_instances))
    print("nb unclassified:", len(unclassified_instances))


# create AI
learner_AI = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model_AI = learner_AI.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, test_size=1 - constants.training_size, seed=123, n_estimators=5) # The same seed


# Create the global theory, enlarge AI in consequence change the representation for user
# Keep the same representation in AI but, increase the binary representation
#model_user => BT
#model_AI => RF
model_user, model_AI = misc.create_binary_representation(model_user, model_AI)


# Create the explainers
explainer_user = Explainer.initialize(model_user, features_type=Tools.Options.types)
explainer_AI = Explainer.initialize(model_AI, features_type=Tools.Options.types)
explainer_AI.set_instance(positive_instances[0])
if constants.debug:
    explainer_user.set_instance(positive_instances[0])
    explainer_AI.set_instance(positive_instances[0])
    assert explainer_user._binary_representation == explainer_AI._binary_representation, "Big problem :)"


# Create the user
user = user.User(explainer_user, positive_instances, negative_instances)

if constants.debug: # Check if all positive and negatives instances are predicted
    for instance in positive_instances:
        explainer_user.set_instance(instance)
        assert(user.predict_instance(explainer_user.binary_representation) != 0)  # we do not take all rules
    for instance in negative_instances:
        explainer_user.set_instance(instance)
        assert(user.predict_instance(explainer_user.binary_representation) != 1)  # we do not take all rules


if constants.trace:
    print("nb positive rules", len(user.positive_rules))
    print("nb negative rules", len(user.negative_rules))
print("WARNING: what about N")


# Iterate on all classified instances

cvg = coverage.Coverage(explainer_AI.get_model().get_theory(explainer_AI.binary_representation), len(explainer_AI.binary_representation), 5, user)
accuracy_user = [user.accurary(classified_instances)]
accuracy_AI = [misc.get_accuracy(explainer_AI.get_model(), classified_instances)]
coverages = [cvg.coverage()]
tmp = 0
nb_instances = 50
for detailed_instance in classified_instances[0:nb_instances]:
    instance = detailed_instance['instance']
    explainer_AI.set_instance(instance)
    prediction_AI = model_AI.predict_instance(instance)
    prediction_user = user.predict_instance(explainer_AI.binary_representation)  # no they have the same representation
    rule_AI = explainer_AI.majoritary_reason(n_iterations=constants.n_iterations)
    # All cases
    print("user: ", prediction_user, "AI: ", prediction_AI)
    if prediction_user is None:  # cases (3) (4) (5)
        match cases.cases_3_4_5(explainer_AI, rule_AI, user):
            case 3:
                constants.statistics["cases_3"] += 1
            case 4:
                constants.statistics["cases_4"] += 1
            case 5:
                constants.statistics["cases_5"] += 1

    else:
        if prediction_AI != prediction_user:  # case (1)
            cases.case_1(explainer_AI, rule_AI, user)
            explainer_AI.set_instance(instance)
            if explainer_AI.target_prediction == prediction_AI:
                print("Aie aie aie")
                print(explainer_AI.binary_representation)
            assert(explainer_AI.target_prediction != prediction_AI)
            constants.statistics["cases_1"] += 1

        if prediction_AI == prediction_user:  # case (2)
            cases.case_2(explainer_AI, rule_AI, user)
            constants.statistics["cases_2"] += 1
    coverages.append(cvg.coverage())
    if constants.trace:
        print(constants.statistics)
        print("------\nnb positive rules", len(user.positive_rules))
        print("nb negative rules", len(user.negative_rules))
        accuracy_AI.append(misc.get_accuracy(explainer_AI.get_model(), test_set=classified_instances))
        accuracy_user.append(user.accurary(classified_instances))
        if len(accuracy_AI) > 1 and accuracy_AI[-1] != accuracy_AI[-2]:
            print("ICCCCCCCC", accuracy_AI[-1], accuracy_AI[-2])
            tmp += 1
        print("accuracy IA", accuracy_AI)
        print("accuracy user: ", accuracy_user)
        print("coverages", coverages)

    #sys.exit(1)
# - accuracy de IA
# - couverture : combien d'instances U est il capable de classer

print("accuracy AI  ", set(accuracy_AI))
print("accuracy user", set(accuracy_user))

print("coverages    ", coverages)































