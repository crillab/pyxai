import user as u


# -------------------------------------------------------------------------------------
def case_1(explainer_AI, rule_AI, user):
    #  Explainer AI and user disagree on the instance (the one hat is in explainer)
    #  Explainer must be rectified
    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules
    nb = 0
    c = 1 if explainer_AI.target_prediction == 0 else 0
    for rule in rules:
        if u.generalize(rule, rule_AI, len(explainer_AI.binary_representation)):
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
