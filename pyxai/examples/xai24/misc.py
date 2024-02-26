import math
import constants
from pyxai import Learning, Explainer, Tools, Builder

#model_user => BT
#model_AI => RF
def create_binary_representation(model_user, model_AI):
    fake_trees = model_AI.forest+model_user.forest
    n_features_max = max(tree.n_features for tree in fake_trees)
    for tree in fake_trees:
        tree.n_features = n_features_max

    
    
    new_model_AI = Builder.RandomForest(fake_trees, n_classes=2, feature_names=model_user.learner_information.feature_names)
    new_model_AI.forest = new_model_AI.forest[0:len(model_AI.forest)]    
    
    new_model_user = Builder.BoostedTrees(fake_trees, n_classes=2, feature_names=model_user.learner_information.feature_names)
    new_model_user.forest = new_model_user.forest[len(model_AI.forest):]    
    
    return new_model_user, new_model_AI

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


