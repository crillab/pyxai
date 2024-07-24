import user as u

import user as u
import constants
import time 

def case_1(explainer_AI, rule_AI, user):
    """
    prediction AI and prediction user differ
    rectify AI with all conflicting rule
    """

    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules
    nb = 0
    rules = user.get_rules_predict_instance(explainer_AI.binary_representation, explainer_AI.target_prediction != 1)
    assert(len(rules) >= 1)
    for rule in rules:
        if True or u.conflict(explainer_AI, rule, rule_AI):
            start_time = time.time()
            explainer_AI.rectify(conditions=rule, label=1 if explainer_AI.target_prediction == 0 else 0)
            nb += 1
            end_time = time.time()
            constants.statistics["rectifications_times"].append(end_time - start_time)
            constants.statistics["rectifications_cases"].append(1)
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
    in_conflict = False
    c = 1 if explainer_AI.target_prediction == 0 else 0
    for rule in rules:
        if u.conflict(explainer_AI, rule, rule_AI):
            in_conflict = True
            start_time = time.time()
            explainer_AI.rectify(conditions=rule, label=c)
            constants.statistics["rectifications"] += 1
            end_time = time.time()
            constants.statistics["rectifications_times"].append(end_time - start_time)
            constants.statistics["rectifications_cases"].append(2)

    # remove specialized rules by rule_AI

    if in_conflict is False:
        # Add rule
        #rules = user.positive_rules if explainer_AI.target_prediction == 1 else user.negative_rules
        #rules.append(rule_AI)

        user.remove_specialized(rule_AI, explainer_AI.target_prediction == 1)
        return 4
    else:
        return 2


# def case_3(explainer_AI, rule_AI, user):
    

# def case_4(explainer_AI, rule_AI, user):
#     """
#     There is no rule in conflict (case 3 does not occur)
#     remove specialized instances
#     """
#     rules = user.positive_rules if explainer_AI.target_prediction == 1 else user.negative_rules
#     correction = False

#     for rule in rules:
#         if u.specialize(explainer_AI, rule_AI, rule):
#             start_time = time.time()
#             explainer_AI.rectify(conditions=rule, label=explainer_AI.prediction)
#             constants.statistics["rectifications"] += 1
#             end_time = time.time()
#             constants.statistics["rectifications_times"].append(end_time - start_time)
#             constants.statistics["rectifications_cases"].append(4)
#             correction = True
#     return correction


# def case_4(explainer_AI, rule_AI, user):
#     """
#     Everything is ok
#     One can add this rule and believe in it
#     """
#     rules = user.positive_rules if explainer_AI.target_prediction == 1 else user.negative_rules
#     rules.append(rule_AI)
#     return True

def cases_3_4(explainer_AI, rule_AI, user):
    
    """
    prediction_user is None
    Policy based
    rectify rules in conflict with prediction
    """
    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules

    c = 1 if explainer_AI.target_prediction == 0 else 0
    correction = False
    for rule in rules:
        if u.conflict(explainer_AI, rule, rule_AI):
            start_time = time.time()
            explainer_AI.rectify(conditions=rule, label=c)
            constants.statistics["rectifications"] += 1
            end_time = time.time()
            constants.statistics["rectifications_times"].append(end_time - start_time)
            constants.statistics["rectifications_cases"].append(3)
            correction = True

    if correction is True:
        return 3
    else:
        # Add rule
        #rules = user.positive_rules if explainer_AI.target_prediction == 1 else user.negative_rules
        #rules.append(rule_AI)

        # Simplify
        user.remove_specialized(rule_AI, explainer_AI.target_prediction == 1)

        return 4
    return correction

    #case_5(explainer_AI, rule_AI, user)
    #return 5

