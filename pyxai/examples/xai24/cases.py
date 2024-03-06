import user as u

import user as u
import constants



def case_1(explainer_AI, rule_AI, user):
    """
    prediction AI and prediction user differ
    rectify AI with all conflicting rule
    """
    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules
    nb = 0
    rules = user.get_rules_predict_instance(explainer_AI.binary_representation, explainer_AI.target_prediction != 1)
    # assert(len(rules) > 1)
    for rule in rules:
        if u.conflict(explainer_AI, rule, rule_AI):
            explainer_AI.rectify(conditions=rule, label=1 if explainer_AI.target_prediction == 0 else 0)
            nb += 1
    if constants.debug:
        assert (nb > 0)  # check that there is no mistake
    constants.statistics["rectifications"] += nb


def case_2(explainer_AI, rule_AI, user):
    """
    User and AI agree
    """
    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules
    #rules2 = user.get_rules_predict_instance(explainer_AI.binary_representation, explainer_AI.target_prediction != 1)
    #assert(len(rules2) == 0)
    # rectify AI with some opposite rules
    c = 1 if explainer_AI.target_prediction == 0 else 0
    for rule in rules:
        if u.conflict(explainer_AI, rule, rule_AI):
            explainer_AI.rectify(conditions=rule, label=c)
            constants.statistics["rectifications"] += 1

    # remove specialized rules by rule_AI
    user.remove_specialized(rule_AI, explainer_AI.target_prediction == 1)



def case_3(explainer_AI, rule_AI, user):
    """
    Policy based
    rectify rules in conflict with prediction
    """
    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules

    c = 1 if explainer_AI.target_prediction == 0 else 0
    correction = False
    for rule in rules:
        if u.conflict(explainer_AI, rule, rule_AI):
            explainer_AI.rectify(conditions=rule, label=c)
            constants.statistics["rectifications"] += 1
            correction = True
    return correction

def case_4(explainer_AI, rule_AI, user):
    """
    There is no rule in conflict (case 3 does not occur)
    remove specialized instances
    """
    rules = user.positive_rules if explainer_AI.target_prediction == 1 else user.negative_rules
    correction = False

    for rule in rules:
        if u.specialize(explainer_AI, rule_AI, rule):
            explainer_AI.rectify(conditions=rule, label=explainer_AI.prediction)
            constants.statistics["rectifications"] += 1
            correction = True
    return correction


def case_5(explainer_AI, rule_AI, user):
    """
    Everything is ok
    One can add this rule and believe in it
    """
    rules = user.positive_rules if explainer_AI.target_prediction == 1 else user.negative_rules
    rules.append(rule_AI)
    return True

def cases_3_4_5(explainer_AI, rule_AI, user):
    """
    User has no idea about the prediction
    """
    if case_3(explainer_AI, rule_AI, user):
        return 3

    if case_4(explainer_AI, rule_AI, user):
        return 4

    case_5(explainer_AI, rule_AI, user)
    return 5

