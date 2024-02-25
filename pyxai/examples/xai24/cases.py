import user as u


# -------------------------------------------------------------------------------------
def case_1(explainer_AI, user):
    #  Explainer AI and user disagree on the instance (the one hat is in explainer)
    #  Explainer must be rectified
    rules = user.positive_rules if explainer_AI.target_prediction == 0 else user.negative_rules
    nb = 0
    for rule in rules:
        if u.generalize(rule, explainer_AI.binary_representation, len(explainer_AI.binary_representation)):
            explainer_AI.rectify(rule, 1 - explainer_AI.target_prediction == 0)  # TODO : classes 0 et 1 ??
            nb += 1
    assert(nb > 0)  # check that there is no mistake



def case_2():
    pass


def cases_3_4_5():
    pass
