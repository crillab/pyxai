from pyxai import Learning, Explainer, Tools
import cases
import user
import constants
import misc



# Create the user agent
learner_user = Learning.Xgboost(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model_user = learner_user.evaluate(method=Learning.HOLD_OUT, output=Learning.BT, test_size=1 - constants.training_size, seed=123)
instances = learner_user.get_instances(model_user, indexes=Learning.TEST, details=True)

# Change weights of BT
if constants.debug:
    print("Accuracy before", misc.get_accuracy(model_user, instances[0:100]))
misc.change_weights(model_user)
if constants.debug:
    print("Accuracy after ", misc.get_accuracy(model_user, instances[0:100]))

# Extract test instances and classified instances
threshold = int(len(instances) * constants.classified_size)
classified_instances = instances[0:threshold]
test_instances = instances[threshold:-1]
positive_instances, negative_instances, unclassified_instances = misc.partition_instances(model_user, classified_instances)

if constants.trace:
    print("nb positives:", len(positive_instances))
    print("nb negatives:", len(negative_instances))
    print("nb unclassified:", len(unclassified_instances))

# Create user
print("WARNING: miss tree specific if sum() > theta or sum() < -theta: to add in Explainer.cc, line 189")
print("WARNING: miss theory")
explainer_user = Explainer.initialize(model_user, features_type=Tools.Options.types)

user = user.User(explainer_user, positive_instances, negative_instances)

if constants.trace:
    print("nb positive rules", len(user.positive_rules))
    print("nb negative rules", len(user.negative_rules))
print("WARNING: what about N")

# create AI
learner_AI = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model_AI = learner_AI.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, test_size=1 - constants.training_size, seed=123) # The same seed
explainer_AI = Explainer.initialize(model_AI, features_type=Tools.Options.types) # miss theory TODO

# Create the global theory, enlarge AI in consequence change the representation for user
# Keep the same representation in AI but, increase the binary representation
sigma = misc.create_binary_representation(explainer_user, explainer_AI)


# Iterate on all classified instances
for detailed_instance in classified_instances:
    instance = detailed_instance['instance']
    explainer_AI.set_instance(instance)

    prediction_AI = model_AI.predict_instance(instance)
    prediction_user = user.predict_instance(explainer_AI.binary_representation) # no they have the same representation
    reason_AI = explainer_AI.majoritary_reason()
    # All cases
    if prediction_user is None:  # cases (3) (4) (5)
        cases.cases_3_4_5()
    else:
        if prediction_AI != prediction_user:  # case (1)
            cases.case1()
        if prediction_AI == prediction_user:  # case (2)
            cases.case2()

# je demande à l'ia
# une des 5 cas arrive et je modifie l'ia en conséquence (et aussi le U)

# on evalue par paquet de 10
# on evalue sur le test set

# - accuracy de IA
# - couverture : combien d'instances U est il capable de classer

































