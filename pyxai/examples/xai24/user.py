import math

import constants


def create_binary_representation(explainer_user, explainer_AI):
    print("numerical user:   ", explainer_user.get_model().map_numerical_features())
    print("categorical user: ", explainer_user.get_model().map_categorical_features_one_hot())
    print("binary user:      ", explainer_user.get_model().map_binary_features())

    return None  # TODO the theory

# -------------------------------------------------------------------------------------
# Change weights of BT and compute accuracy of a test set

def get_accuracy(model, test_set):
    nb = 0
    for instance in test_set:
        prediction = model.predict_instance(instance["instance"])
        nb += 1 if prediction == instance['label'] else 0
    return nb / len(test_set)


def maximum_weight(model):
    max_weight = -math.inf
    for tree in model.forest:
        leaves = tree.get_leaves()
        max_weight = max(max_weight, max([abs(leave.value) for leave in leaves]))
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
        score = 0
        for tree in model.forest:
            score += tree.predict_instance(instance)
        if score > constants.theta:
            positive.append(instance)
        elif score < -constants.theta:
            negative.append(instance)
        else:
            unclassified.append(instance)
    return positive, negative, unclassified


# -------------------------------------------------------------------------------------


def generalize(rule1, rule2, len_binary):
    """
    Return True if rule1 generalizes rule2
    a generalize ab
    """
    if len(rule1) > len(rule2):
        return False

    occurences = [0 for _ in range(len_binary + 1)]
    for lit in rule1:
        occurences[abs(lit)] = lit
    for lit in rule2:
        if occurences[abs(lit)] != lit:
            return False
    return True


def is_classified_by_user(binary_representation):
    """
    Take in parameter the binary representation of an instance
    return 1 if it is classified 1
    return 0 if it is classified 0
    return None otherwise
    """
    for rule in positive_rules:
        if generalize(rule, binary_representation, len(binary_representation)):
            return 1
    for rule in negative_rules:
        if generalize(rule, binary_representation, len(binary_representation)):
            return 0
    return None


# -------------------------------------------------------------------------------------
# Create the rules for a given set of instances
def create_rules(explainer, instances):
    result = []
    for instance in instances:
        explainer.set_instance(instance)
        reason = explainer.tree_specific_reason(n_iterations=1)
        new_rule = True
        for rule in result:  # reason does not specialize existing rule
            if generalize(rule, reason, len(explainer.binary_representation)):
                # print("\n---\nreason:", reason, "\nspecial:", rule)
                new_rule = False
                break
        if new_rule:  # if not
            tmp = []  # can be done more efficiently
            for rule in result:  # remove specialied rules
                if not generalize(reason, rule, len(explainer.binary_representation)):
                    tmp.append(rule)
                else:
                    pass
                    # print("\n---\nrule:", rule, "\nspecial:", reason)

            tmp.append(reason)  # do not forget to add this one
            result = tmp
    return result
