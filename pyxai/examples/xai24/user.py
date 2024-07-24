from pyxai import Explainer, Learning, Tools
import constants
import random
import misc


class User:
    def __init__(self, explainer, positive_instances, negative_instances):
        self.explainer = explainer
        explainer.set_instance(positive_instances[0])
        self.nb_variables = len(explainer.binary_representation)
        n_total = len(positive_instances) + len(negative_instances)
        n_positives = round((len(positive_instances) / n_total) * constants.N)
        n_negatives = round((len(negative_instances) / n_total) * constants.N)
        random.shuffle(positive_instances)
        random.shuffle(negative_instances)

        self.positive_rules = self._create_rules(explainer, positive_instances, constants.theta, n_positives)
        self.negative_rules = self._create_rules(explainer, negative_instances, -constants.theta, n_negatives)

    def predict_instance(self, binary_representation):
        """
        Take in parameter the binary representation of an instance
        return 1 if it is classified 1
        return 0 if it is classified 0
        return None otherwise
        """
        for rule in self.positive_rules:
            if generalize(self.explainer, rule, binary_representation):
                return 1
        for rule in self.negative_rules:
            if generalize(self.explainer, rule, binary_representation):
                return 0
        return None

    def get_rules_predict_instance(self, binary_representation, prediction):
        tmp = []
        if prediction:
            for rule in self.positive_rules:
                if generalize(self.explainer, rule, binary_representation):
                    tmp.append(rule)
        else:
            for rule in self.negative_rules:
                if generalize(self.explainer, rule, binary_representation):
                    tmp.append(rule)
        return tmp

    def remove_specialized(self, reason, positive):

        
        rules = self.positive_rules if positive else self.negative_rules
        
        tmp = [r for r in rules if len(reason) == 0 or not generalize(self.explainer, reason, r)]
        
        if len(tmp) != len(rules):
            tmp.append(reason)
            constants.statistics["generalisations"] += 1
            if positive:
                self.positive_rules = tmp
            else:
                self.negative_rules = tmp

    # -------------------------------------------------------------------------------------
    #  Create the rules for a given set of instances
    def _create_rules(self, explainer, instances, theta, nb):
        result = []
        for instance in instances:
            explainer.set_instance(instance)

            reason = explainer.tree_specific_reason(n_iterations=constants.n_iterations, theta=theta)

            if is_really_new_rule(self.explainer, result, reason):  # if not
                result = remove_all_specialized(self.explainer, result, reason)


                if len(result) == nb:
                    break
        return result

    def accurary(self, test_set):
        nb = 0
        total = 0
        for instance in test_set:
            self.explainer.set_instance(instance["instance"])
            prediction = self.predict_instance(self.explainer.binary_representation)
            #print("prediction:", prediction)
            if prediction is not None:
                nb += 1 if prediction == instance['label'] else 0
                total += 1
        if total == 0:
            return None
        return nb / total



class UserLambda(User):
    def __init__(self, explainer, nb_v, positive_rules, negative_rules):
        self.explainer = explainer
        self.nb_variables = nb_v
        print("self.nb_variables user:", self.nb_variables)
        self.positive_rules = positive_rules
        self.negative_rules = negative_rules


# -------------------------------------------------------------------------------------

# c statistics {'rectifications': 21, 'generalisations': 13, 'cases_1': 5, 'cases_2': 20, 'cases_3': 0, 'cases_4': 0, 'cases_5': 5, 'n_positive': 0, 'n_negatives': 113, 'n_positives': 74}

def generalize(explainer_AI, rule1, rule2):
    """
    Return True if rule1 generalizes rule2
    a generalize ab
    """
    tmp1 = explainer_AI.extend_reason_with_theory(rule1)
    tmp2 = explainer_AI.extend_reason_with_theory(rule2)
    
    if len(tmp1) > len(tmp2):
        return False

    occurrences = {}
    for lit in tmp2:
        occurrences[lit] = 1

    for lit in tmp1:
        if occurrences.get(lit) is None:
            return False

    # occurences = [0 for _ in range(len_binary + 1)]
    # for lit in rule1:
    #    occurences[abs(lit)] = lit
    # for lit in rule2:
    #    if occurences[abs(lit)] != lit:
    #        return False
    return True


def is_really_new_rule(explainer, rules, new_rule):
    for rule in rules:  # reason does not specialize existing rule
        if generalize(explainer, rule, new_rule):
            return False
    return True


def remove_all_specialized(explainer, rules, reason):
    tmp = []  # can be done more efficiently
    for rule in rules:  # remove specialized rules
        if not generalize(explainer, reason, rule):
            tmp.append(rule)
        else:
            pass
        # print("\n---\nrule:", rule, "\nspecial:", reason)

    tmp.append(reason)  # do not forget to add this one
    return tmp



def specialize(explainer_AI, rule1, rule2):
    return generalize(explainer_AI, rule2, rule1)


from pysat.solvers import Glucose4

gluglu = None


def conflict(explainer, rule1, rule2):
    """
    Check if two rules are in conflict
    """
    global gluglu
    if gluglu is None:
        gluglu = Glucose4()
        gluglu.append_formula(explainer.get_model().get_theory([]))  # no need of binary representation

    return gluglu.solve(assumptions=rule1 + rule2)  # conflict if SAT

    """
    Old version
    tmp1 = explainer_AI.extend_reason_with_theory(rule1)
    tmp2 = explainer_AI.extend_reason_with_theory(rule2)
    for lit in tmp1:
        if -lit in tmp2:
            return False
    return True
"""


def create_user_BT(AI):
    # Create the user agent
    print("create BT")
    learner_user = Learning.Xgboost(Tools.Options.dataset, learner_type=Learning.CLASSIFICATION)
    model_user = learner_user.evaluate(method=Learning.HOLD_OUT, output=Learning.BT,
                                       test_size=1 - constants.training_size, seed=123)
    instances = learner_user.get_instances(model_user, indexes=Learning.TEST, details=True)
    # Change weights of BT
    misc.change_weights(model_user)

    # Extract test instances and classified instances
    threshold = int(len(instances) * constants.classified_size)
    classified_instances = instances[0:threshold]
    positive_instances, negative_instances, unclassified_instances = misc.partition_instances(model_user,
                                                                                              classified_instances)

    if constants.trace:
        print("nb positives:", len(positive_instances))
        print("nb negatives:", len(negative_instances))
        print("nb unclassified:", len(unclassified_instances))

    # Create the global theory, enlarge AI in consequence change the representation for user
    # Keep the same representation in AI but, increase the binary representation
    # model_user => BT
    # model_AI => RF / DT
    model_user, AI.model = misc.create_binary_representation(model_user, AI)

    # Create the explainers
    explainer_user = Explainer.initialize(model_user, features_type=Tools.Options.types)
    AI.explainer = Explainer.initialize(AI.model, features_type=Tools.Options.types)
    AI.set_instance(positive_instances[0])
    explainer_user.set_instance(positive_instances[0])
    if constants.debug:
        AI.set_instance(positive_instances[0])
        assert explainer_user._binary_representation == AI.explainer._binary_representation, "Big problem :)"

    # Create the user
    print("Create user")
    user = User(explainer_user, positive_instances, negative_instances)

    if constants.debug:  # Check if all positive and negatives instances are predicted
        for instance in positive_instances:
            explainer_user.set_instance(instance)
            assert (user.predict_instance(explainer_user.binary_representation) != 0)  # we do not take all rules
        for instance in negative_instances:
            explainer_user.set_instance(instance)
            assert (user.predict_instance(explainer_user.binary_representation) != 1)  # we do not take all rules
    return user

def create_user_lambda_forest(AI, classified_instances):
    positive_rules = []
    negative_rules = []
    random.seed(123)
    print("type:", type(AI))

    #lenght_reasons = []
    #ff= []
    for i, detailed_instance in enumerate(classified_instances):
        if len(positive_rules) + len(negative_rules) >= constants.N:
            break
        #AI.set_instance(detailed_instance["instance"])
        votes, prediction = AI.model.predict_votes(detailed_instance["instance"])
        tmp0 = votes[0]/(votes[0]+votes[1])
        tmp1 = votes[1]/(votes[0]+votes[1])
        votes = tmp0, tmp1
        if (prediction == 0 and tmp0 >= constants.delta) or (prediction == 1 and tmp1 >= constants.delta):
        
            AI.set_instance(detailed_instance["instance"])
            rule = AI.reason(n_iterations=50)
            
            if len(rule) == len(AI.explainer.binary_representation):
                continue
            result = positive_rules if prediction == 1 else negative_rules
            if is_really_new_rule(AI.explainer, result, rule):  # if not
                tmp = remove_all_specialized(AI.explainer, result, rule)

                #lenght_reasons.append(len(rule))
                #ff.append(len(AI.explainer.to_features(rule)))
                if prediction == 1:
                    positive_rules = tmp
                else:
                    negative_rules = tmp
    #lenght_reasons = sum(lenght_reasons)/len(lenght_reasons)
    #print("avg:", lenght_reasons)
    #print("len(AI.explainer.binary_representation):", len(AI.explainer.binary_representation))
    
    #lenght_reasons = lenght_reasons/len(AI.explainer.binary_representation)
    #print("lenght_reasons:", lenght_reasons)
    #exit(0)
    #AI.set_instance(classified_instances[0]["instance"])
    #print(positive_rules)
    #print(negative_rules)
    """
    for rule1 in positive_rules:
        for rule2 in negative_rules:
            assert(conflict(AI.explainer, rule1, rule2) is False)
    for rule1 in negative_rules:
        for rule2 in positive_rules:
            assert (conflict(AI.explainer, rule1, rule2) is False)
    """
    user = UserLambda(AI.explainer, len(AI.explainer.binary_representation), positive_rules, negative_rules)
    return user

def create_user_lambda(AI, classified_instances):
    positive_rules = []
    negative_rules = []
    random.seed(123)
    
    for detailed_instance in classified_instances:
        if len(positive_rules) + len(negative_rules) >= constants.N:
            break
        AI.set_instance(detailed_instance["instance"])
        rule = AI.reason(n_iterations=1)
        if len(rule) == len(AI.explainer.binary_representation) or  random.randint(0, 2) == 0:  # 1/3 not classified
            continue

        # remove a literal from the rule?
        if random.randint(0, 1) == 0:
            list(rule).pop()

        # good or wrong classification
        classification =random.randint(0, 4)
        if classification == 0: # wrong
            rules = positive_rules if AI.explainer.target_prediction == 0 else negative_rules
        if classification == 1:  # good wrt AI
            rules = positive_rules if AI.explainer.target_prediction == 1 else negative_rules
        if classification > 1:  # good wrt dataset
            rules = positive_rules if detailed_instance["label"] == 1 else negative_rules

        rules.append(rule)
    AI.set_instance(classified_instances[0]["instance"])
    user = UserLambda(AI.explainer, len(AI.explainer.binary_representation), positive_rules, negative_rules)

    return user



