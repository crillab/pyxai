import math
import constants
class User:
    def __init__(self, explainer, positive_instances, negative_instances):
        self.positive_rules = self._create_rules(explainer, positive_instances, constants.theta)
        self.negative_rules = self._create_rules(explainer, negative_instances, -constants.theta)

    def predict_instance(self, binary_representation):
        """
        Take in parameter the binary representation of an instance
        return 1 if it is classified 1
        return 0 if it is classified 0
        return None otherwise
        """
        for rule in self.positive_rules:
            if generalize(rule, binary_representation, len(binary_representation)):
                return 1
        for rule in self.negative_rules:
            if generalize(rule, binary_representation, len(binary_representation)):
                return 0
        return None


    # -------------------------------------------------------------------------------------
    #  Create the rules for a given set of instances
    def _create_rules(self, explainer, instances, theta):
        result = []
        for instance in instances:
            explainer.set_instance(instance)
            reason = explainer.tree_specific_reason(n_iterations=1, theta=theta)
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



