import math

import constants

def create_binary_representation(explainer_user, explainer_AI):
    for key in explainer_user.get_model().map_numerical_features.keys():
        id_binaries = explainer_user.get_model().map_numerical_features[key]
        conditions = [tuple(list(explainer_user.get_model().map_id_binaries_to_features[id]) + [id]) for id in id_binaries]
        conditions = sorted(conditions, key=lambda t: t[2], reverse=True)
        id_binaries_sorted = tuple(condition[3] for condition in conditions)
        print(conditions)
        #id_binaries_sorted = tuple(condition[3] for condition in conditions)

    #print("numerical user:   ", explainer_user.get_model().map_numerical_features)
    #print("categorical user: ", explainer_user.get_model().map_categorical_features_one_hot)
    #print("binary user:      ", explainer_user.get_model().map_binary_features)

    return 0, 0  # TODO the theory the number of variables

#------------------------------------------------------------------------------------
# Change weights of BT and compute accuracy of a test set

def get_accuracy(model, test_set):
    nb = 0
    for instance in test_set:
        prediction = model.predict_instance(instance["instance"])
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


