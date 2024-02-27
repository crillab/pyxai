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
    for rule in rules:
        if u.conflict(rule, rule_AI):
            explainer_AI.rectify(rule, 1 if explainer_AI.target_prediction == 0 else 0)
            nb += 1
    if constants.debug:
        assert (nb > 0)  # check that there is no mistake


def case_2(explainer_AI, rule_AI, user):
    """
    User and AI agree
    """
    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules

    # rectify AI with some opposoite rules
    c = 1 if explainer_AI.target_prediction == 0 else 0
    for rule in rules:
        if u.conflict(rule, rule_AI):
            explainer_AI.rectify(rule, c)

    # remove specialized rules by rule_AI
    user.remove_specialized(rule_AI, explainer_AI.target_prediction == 1)




def case_3(explainer_AI, rule_AI, user):
    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules

    c = 1 if explainer_AI.target_prediction == 0 else 0
    correction = False
    for rule in rules:
        if u.conflict(rule, rule_AI):
            explainer_AI.rectify(rule, c)
            correction = True
    return correction

def case_4(explainer_AI, rule_AI, user):
    rules = user.positive_rules if explainer_AI.target_prediction == 1 else user.negative_rules
    correction = False

    for rule in rules:
        if u.specialize(rule_AI, rule):
            explainer_AI.rectify(rule, explainer_AI.prediction)  # TODO : classes 0 et 1 ??
            correction = True

    return correction


def case_5(explainer_AI, rule_AI, user):

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


"""
# -------------------------------------------------------------------------------------
def case_1(explainer_AI, rule_AI, user):
    #  Explainer AI and user disagree on the instance (the one hat is in explainer)
    #  Explainer must be rectified
    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules
    nb = 0
    c = 1 if explainer_AI.target_prediction == 0 else 0
    for rule in rules:
        if u.generalize(rule, rule_AI)):
            explainer_AI.rectify(rule, c)  # TODO : classes 0 et 1 ??
            nb += 1
    assert (nb > 0)  # check that there is no mistake


def case_2(explainer_AI, rule_AI, user):
    # Explainer AI and user agree on the instance

    #  Step 1 rectify the other rules
    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules
    c = 1 if explainer_AI.target_prediction == 0 else 0
    for rule in rules:
        if u.generalize(rule, rule_AI, len(explainer_AI.binary_representation)): # TODO FAUX
            explainer_AI.rectify(rule, c)  # TODO : classes 0 et 1 ??

    #  Step 2 : remove rules that are generalized by AI rule
    user.remove_specialized(rule_AI, explainer_AI.target_prediction == 1)


def cases_3_4_5(explainer_AI, rule_AI, user):
    pass
"""
