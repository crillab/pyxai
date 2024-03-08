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
if Tools.Options.n is not None:
    constants.N = int(Tools.Options.n)
print("N = ", constants.N)
assert(constants.model == Learning.RF or constants.model == Learning.DT)


Tools.set_verbose(0)
# Create the user agent
print("create BT")
learner_user = Learning.Xgboost(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
model_user = learner_user.evaluate(method=Learning.HOLD_OUT, output=Learning.BT, test_size=1 - constants.training_size, seed=123)
instances = learner_user.get_instances(model_user, indexes=Learning.TEST, details=True)
# Change weights of BT
misc.change_weights(model_user)


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
print("Create AI: ", "DT" if constants.model == Learning.DT else "RF")
learner_AI = Learning.Scikitlearn(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
AI = model.Model(learner_AI)

# Create the global theory, enlarge AI in consequence change the representation for user
# Keep the same representation in AI but, increase the binary representation
#model_user => BT
#model_AI => RF / DT
model_user, AI.model = misc.create_binary_representation(model_user, AI)



# Create the explainers
explainer_user = Explainer.initialize(model_user, features_type=Tools.Options.types)
AI.explainer = Explainer.initialize(AI.model, features_type=Tools.Options.types)
AI.set_instance(positive_instances[0])
explainer_user.set_instance(positive_instances[0])
if constants.debug:
    AI.set_instance(positive_instances[0])
    assert explainer_user._binary_representation == AI.explainer._binary_representation, "Big problem :)"


# Create the user
print("Create user")
user = user.User(explainer_user, positive_instances, negative_instances)

if constants.debug:  # Check if all positive and negatives instances are predicted
    for instance in positive_instances:
        explainer_user.set_instance(instance)
        assert(user.predict_instance(explainer_user.binary_representation) != 0)  # we do not take all rules
    for instance in negative_instances:
        explainer_user.set_instance(instance)
        assert(user.predict_instance(explainer_user.binary_representation) != 1)  # we do not take all rules


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
    tmp = explainer_user.to_features(rule,details=True, eliminate_redundant_features=False)
    for key in tmp.keys():
        if key not in dict_features:
            dict_features[key] = 1
    nb_features.append(len(dict_features.keys()))
for rule in user.negative_rules:
    dict_features = {}
    tmp = explainer_user.to_features(rule,details=True, eliminate_redundant_features=False)
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
    tmp = explainer_user.to_features(rule,details=True, eliminate_redundant_features=False)
    for key in tmp.keys():
        if key not in dict_features:
            dict_features[key] = 1
    nb_features.append(len(dict_features.keys()))
for rule in user.negative_rules:
    dict_features = {}
    tmp = explainer_user.to_features(rule,details=True, eliminate_redundant_features=False)
    for key in tmp.keys():
        if key not in dict_features:
            dict_features[key] = 1
    nb_features.append(len(dict_features.keys()))

print("c nb features at end:", nb_features)



"""
jeu_de_donnée='compas'
epochs = list(range(1, len(accuracy_AI_user) + 1))

# Créer le graphique
plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracy_AI_user, marker='o', color='skyblue', linestyle='-')

# Ajouter des titres et des labels
plt.title('accuracy_AI ')
plt.xlabel('nb rules')
plt.ylabel('accuracy_AI')

# Définir les intervalles pour les axes
plt.xticks(range(1, len(accuracy_AI_user) + 1, 6))
# Afficher le graphique
plt.grid(True)
plt.savefig(jeu_de_donnée+'_accuracy_AI.png')

plt.show()


epochs = list(range(1, len(accuracy_user) + 1))

# Créer le graphique
plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracy_user, marker='o', color='skyblue')

# Ajouter des titres et des labels
plt.title('accuracy_user ')
plt.xlabel('nb rules')
plt.ylabel('accuracy_user')

# Définir les intervalles pour les axes
plt.xticks(range(1, len(accuracy_user) + 1, 6))
# Afficher le graphique
plt.grid(True)
plt.savefig(jeu_de_donnée+'_accuracy_user.png')

plt.show()



# Créer le graphique
plt.figure(figsize=(8, 6))
plt.plot(epochs, coverages, marker='o', color='skyblue')

# Ajouter des titres et des labels
plt.title('coverages ')
plt.xlabel('nb rules')
plt.ylabel('coverages')

# Définir les intervalles pour les axes
plt.xticks(range(1, len(coverages) + 1, 6))
# Afficher le graphique
plt.grid(True)
plt.savefig(jeu_de_donnée+'_coverages.png')

plt.show()




# Données pour accuracy_AI
epochs_AI = list(range(1, len(accuracy_AI_user) + 1))

# Créer le graphique pour accuracy_AI
plt.figure(figsize=(10, 6))
plt.plot(epochs_AI, accuracy_AI_user, marker='o', color='skyblue', label='accuracy_AI')

# Données pour accuracy_user
epochs_user = list(range(1, len(accuracy_user) + 1))

# Ajouter le graphique pour accuracy_user dans le même graphique
plt.plot(epochs_user, accuracy_user, marker='o', color='orange', label='accuracy_user')

# Ajouter des titres et des labels
plt.title('Accuracy Comparison')
plt.xlabel('nb rules')
plt.ylabel('Accuracy')
plt.legend()  # Ajouter la légende

# Définir les intervalles pour les axes
plt.xticks(range(1, max(len(accuracy_AI_user), len(accuracy_user)) + 1, 6))

# Afficher la grille
plt.grid(True)

# Sauvegarder le graphique au format PNG
plt.savefig(jeu_de_donnée+'_accuracy_user_and_accIA.png')

# Afficher le graphique
plt.show()
"""