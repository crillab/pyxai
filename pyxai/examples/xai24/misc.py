import math
import constants
from pyxai import Learning, Explainer, Tools, Builder

#model_user => BT
#model_AI => RF


def create_binary_representation_DT(model_user, AI):
    model_AI = AI.model
    fake_trees = [model_AI] + model_user.forest
    n_features_max = max(tree.n_features for tree in fake_trees)
    for tree in fake_trees:
        tree.n_features = n_features_max

    new_model_AI = Builder.RandomForest(fake_trees, n_classes=2,
                                        feature_names=model_user.learner_information.feature_names)
    model_AI = new_model_AI.forest[0]
    model_AI.map_id_binaries_to_features = new_model_AI.map_id_binaries_to_features
    model_AI.map_features_to_id_binaries = new_model_AI.map_features_to_id_binaries


    new_model_user = Builder.BoostedTrees(fake_trees, n_classes=2,
                                          feature_names=model_user.learner_information.feature_names)
    new_model_user.forest = new_model_user.forest[1:]

    return new_model_user, model_AI


def create_binary_representation_RF(model_user, AI):
    model_AI = AI.model
    fake_trees = model_AI.forest+model_user.forest
    n_features_max = max(tree.n_features for tree in fake_trees)
    for tree in fake_trees:
        tree.n_features = n_features_max

    
    
    new_model_AI = Builder.RandomForest(fake_trees, n_classes=2, feature_names=model_user.learner_information.feature_names)
    new_model_AI.forest = new_model_AI.forest[0:len(model_AI.forest)]    
    
    new_model_user = Builder.BoostedTrees(fake_trees, n_classes=2, feature_names=model_user.learner_information.feature_names)
    new_model_user.forest = new_model_user.forest[len(model_AI.forest):]    
    
    return new_model_user, new_model_AI

def create_binary_representation(model_user, AI):
    if constants.model == Learning.DT:
        return create_binary_representation_DT(model_user, AI)
    else:
        return create_binary_representation_RF(model_user, AI)


#------------------------------------------------------------------------------------
# Change weights of BT and compute accuracy of a test set

def get_accuracy(model, test_set):
    nb = 0
    for instance in test_set:
        prediction = model.predict_instance(instance["instance"])
        nb += 1 if prediction == instance['label'] else 0
    return nb / len(test_set)

def get_accuracy_bin(explainer, test_set):
    nb = 0
    for instance in test_set:
        explainer.set_instance(instance["instance"])
        prediction = explainer.get_model().predict_implicant(explainer.binary_representation)
        nb += 1 if prediction == instance['label'] else 0
    return nb / len(test_set)


def maximum_weight(model):
    max_weight = max((abs(leave.value) for tree in model.forest for leave in tree.get_leaves()))
    return max_weight


def change_weights(model):
    max_weight = maximum_weight(model)
    for tree in model.forest:
        leaves = tree.get_leaves()
        for leave in leaves:
            leave.value = leave.value / max_weight

def acuracy_wrt_user(user, explainer_AI, model_AI, test_set) : 
    nb = 0
    for instance in test_set:
        explainer_AI.set_instance(instance["instance"])

        prediction_user = user.predict_instance(explainer_AI.binary_representation)
        if prediction_user is not None:
            if model_AI.predict_instance(instance["instance"]) == prediction_user:
                nb += 1
        else : 
            if model_AI.predict_instance(instance["instance"]) == instance["label"]:
                nb += 1
    return nb / len(test_set)
# -------------------------------------------------------------------------------------
# Partition instances between positive, negative and unclassified ones (wrt BT model (user)
def partition_instances(model, classified_instances):
    positive = []
    negative = []
    unclassified = []
    for detailed_instance in classified_instances:
        instance = detailed_instance["instance"]
        score = sum((tree.predict_instance(instance) for tree in model.forest))
        if score > constants.theta:
            positive.append(instance)
        elif score < -constants.theta:
            negative.append(instance)
        else:
            unclassified.append(instance)
    return positive, negative, unclassified


def print_features(user):
    nb_binaries = [len(rule) for rule in user.positive_rules] + [len(rule) for rule in user.negative_rules]
    print("c nb binaries at start:", nb_binaries)

    nb_features = []
    for rule in user.positive_rules:
        dict_features = {}
        tmp = user.explainer.to_features(rule, details=True, eliminate_redundant_features=False)
        for key in tmp.keys():
            if key not in dict_features:
                dict_features[key] = 1
        nb_features.append(len(dict_features.keys()))
    for rule in user.negative_rules:
        dict_features = {}
        tmp = user.explainer.to_features(rule, details=True, eliminate_redundant_features=False)
        for key in tmp.keys():
            if key not in dict_features:
                dict_features[key] = 1
        nb_features.append(len(dict_features.keys()))

    print("c nb features at start:", nb_features)

