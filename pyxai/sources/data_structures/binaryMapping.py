from pyxai.sources.type import OperatorCondition, TypeTheory
from pyxai.sources.tools.encoding import CNFencoding


import pandas
from numpy import argmax, argmin
import collections
from collections.abc import Iterable

class BinaryMapping():

    def __init__(self, map_id_binaries_to_features, map_features_to_id_binaries):
        self.map_id_binaries_to_features = map_id_binaries_to_features
        self.map_features_to_id_binaries = map_features_to_id_binaries
        self.map_numerical_features = {}  # dict[id_feature] -> [id_binaries of the feature]
        self.map_binary_features = {}  # dict[id_feature] -> [id_binaries of the feature]
        self.map_categorical_features_ordinal = {}  # dict[id_feature] -> [id_binaries of the feature]
        self.map_categorical_features_one_hot = {}  # dict[reg_exp_name] -> [id_binaries of the set of features representing the reg_exp_name of the feature that was one hot encoded]
        self.values_categorical_features = {} # dict[feature name]->categorical value 
        self.map_check_already_used = {}  # Just to check if a feature is already used


        self.n_redundant_features = 0  # Variable to store the number of redondances in eliminate_redundant_features()

    def n_binary_variables(self):
        return len(self.map_id_binaries_to_features)-1

    



    @property
    def numerical_features(self):
        return self.map_numerical_features


    @property
    def categorical_features_ordinal(self):
        return self.map_categorical_features_ordinal


    @property
    def categorical_features_one_hot(self):
        return self.map_categorical_features_ordinal


    def get_used_features(self):
        used_features = set()
        for key in self.map_features_to_id_binaries.keys():
            used_features.add(key[0])
        return len(used_features)


    def clear_theory_features(self):
        self.map_check_already_used.clear()
        self.map_numerical_features.clear()
        self.map_binary_features.clear()
        self.map_categorical_features_one_hot.clear()
        self.map_categorical_features_ordinal.clear()
        self.values_categorical_features.clear()

    def add_numerical_feature(self, id_feature):
        if id_feature in self.map_check_already_used.keys():
            raise ValueError("The feature id (" + str(id_feature) + ") is already used as" + self.map_check_already_used[id_feature])

        id_binaries_of_the_feature = []
        for key in self.map_features_to_id_binaries.keys():
            if key[0] == id_feature:
                id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])

        self.map_numerical_features[id_feature] = id_binaries_of_the_feature
        self.map_check_already_used[id_feature] = "numerical"


    def add_binary_feature(self, id_feature):
        if id_feature in self.map_check_already_used.keys():
            raise ValueError("The feature id (" + str(id_feature) + ") is already used as" + self.map_check_already_used[id_feature])

        id_binaries_of_the_feature = []
        for key in self.map_features_to_id_binaries.keys():
            if key[0] == id_feature:
                id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])

        self.map_binary_features[id_feature] = id_binaries_of_the_feature
        self.map_check_already_used[id_feature] = "binary"


    def set_values_categorical_features(self, values_categorical_features):
        self.values_categorical_features = values_categorical_features

    def add_categorical_feature_one_hot(self, reg_exp_feature_name, id_features):
        id_binaries_of_the_feature = []
        for id_feature in id_features:
            if id_feature in self.map_check_already_used.keys():
                raise ValueError("The feature id (" + str(id_feature) + ") is already used as " + self.map_check_already_used[id_feature])

        for key in self.map_features_to_id_binaries.keys():
            for id_feature in id_features:
                if key[0] == id_feature:
                    id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])

        self.map_categorical_features_one_hot[reg_exp_feature_name] = id_binaries_of_the_feature
        for id_feature in id_features:
            self.map_check_already_used[id_feature] = "categorical"


    def add_categorical_feature_ordinal(self, id_feature):
        raise NotImplementedError  # TO DO See with Gilles ? on laisse pour l'instant
        if id_feature in self.map_numerical_features:
            raise ValueError("The given id_feature (" + str(id_feature) + ") is already considered as numerical.")
        if id_feature in self.map_categorical_features_ordinal:
            raise ValueError("The given id_feature (" + str(id_feature) + ") is already considered as categorical ordinal.")

        id_binaries_of_the_feature = []
        for key in self.map_features_to_id_binaries.keys():
            if key[0] == id_feature:
                id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])

        self.map_categorical_features_ordinal[id_feature] = id_binaries_of_the_feature

    #
    # Check the conditions and return the conditions in the form of literals
    #
    def parse_conditions_for_rectify(self, conditions):
        new_conditions = []
        change = False
                    
        for element in conditions:
            if isinstance(element, int):
                new_conditions.append(element)
            elif isinstance(element, tuple):
                if len(element) != 4:
                    raise ValueError("A condition in rectify is either a literal representing a condition or a condition in the form (id_feature, operator, value, sign).")
                condition = tuple(element[0:3])
                sign = element[3]
                if condition in self.map_features_to_id_binaries:
                    #Nothing to do: the literal exists.
                    if sign == True:
                        new_conditions.append(-self.map_features_to_id_binaries[condition][0])
                    else:
                        new_conditions.append(self.map_features_to_id_binaries[condition][0])
                        
                else:
                    #The literal does not exist
                    id_binary = len(self.map_id_binaries_to_features)
                    if not isinstance(condition[1], OperatorCondition):
                            raise ValueError("A condition in rectify is either a literal representing a condition or a condition in the form (id_feature, operator, value, sign). The operator have to be a OperatorCondition object.")

                    self.map_features_to_id_binaries[condition] = [id_binary, 0, 0]
                    self.map_id_binaries_to_features.append(condition)
                    assert self.map_id_binaries_to_features[id_binary] == condition
                    new_conditions.append(-id_binary if sign == True else id_binary)
                    change = True
            else:
                raise ValueError("A condition in rectify is either a literal (int) representing a condition or a condition in the form (id_feature, operator, value, sign).")
        return tuple(new_conditions), change      
                
                
    def get_theory(self, binary_representation, *, theory_type=TypeTheory.Simple, id_new_var=0):
        
        if theory_type == TypeTheory.NewVariables:
            if binary_representation is None:
                raise ValueError("The `binary_representation` parameter cannot be None with theory_type=TypeTheory.NewVariables")
            # structure to help to do this method faster
            map_id_binary_sign = dict()
            map_is_represented_by_new_variables = dict()
            for id in binary_representation:
                map_id_binary_sign[abs(id)] = 1 if id > 0 else -1
                map_is_represented_by_new_variables[abs(id)] = False

        clauses = []
        new_variables = []

        # For numerical features
        for key in self.map_numerical_features.keys():
            id_binaries = self.map_numerical_features[key]
            conditions = [tuple(list(self.map_id_binaries_to_features[id]) + [id]) for id in id_binaries]
            conditions = sorted(conditions, key=lambda t: t[2], reverse=True)
            id_binaries_sorted = tuple(condition[3] for condition in conditions)

            for i in range(len(id_binaries_sorted) - 1):  # To not takes the last
                clauses.append((-id_binaries_sorted[i], id_binaries_sorted[i + 1]))

            if theory_type == TypeTheory.NewVariables:
                id_new_var = id_new_var + 1
                # associated_literals = []
                for id_binary in id_binaries_sorted:
                    clauses.append((-id_new_var, map_id_binary_sign[id_binary] * id_binary))
                    map_is_represented_by_new_variables[id_binary] = True
                    # associated_literals.append(map_id_binary_sign[id_binary]*id_binary)
                new_variables.append(id_new_var)

        # For categorical features that was one hot encoded
        for key in self.map_categorical_features_one_hot.keys():
            id_binaries = self.map_categorical_features_one_hot[key]
            
            for i, id_1 in enumerate(id_binaries):
                for j, id_2 in enumerate(id_binaries):
                    if i != j:
                        #if self.map_id_binaries_to_features[abs(id_1)][0] == self.map_id_binaries_to_features[abs(id_2)][0]:
                        #    clauses.append((-id_1, id_2))
                        #    clauses.append((id_1, -id_2))
                        #else:
                            # we code a => not b that is equivalent to not a or not b (material implication)
                        clauses.append((-id_1, -id_2))
                            # For binary feature, nothing to do.

        if theory_type == TypeTheory.Simple:
            return clauses
        elif theory_type == TypeTheory.NewVariables:
            return clauses, (new_variables, map_is_represented_by_new_variables)
        else:
            raise NotImplementedError


    def _compute_id_binaries(self):
        assert False, "Have to be implemented in a child class."


    def instance_to_binaries(self, instance, preference_order=None):
        """
        map_id_binaries_to_features: list[id_binary] -> (id_feature, operator, threshold)
        map_features_to_id_binaries: dict[(id_feature, operator, threshold)] -> [id_binary, n_appears, n_appears_per_tree]
        Transform a instance into a cube (conjunction of literals) according to the tree
        """
        output = []
        if isinstance(instance, pandas.Series) and 'subset' in instance.index:
            instance = instance.drop(index='subset')
    
        instance = self.learner_information.get_item_function(instance)

        instance = pandas.Series(instance)
        if preference_order is None:
            for key in self.map_features_to_id_binaries.keys():  # the keys are of the type: (id_feature, operator, threshold)
                id_feature = key[0]
                operator = key[1]
                threshold = key[2]

                if instance.iloc[id_feature - 1] is None:  # It is possible to compute a partial binary representation
                    continue

                if operator == OperatorCondition.GE:
                    sign = 1 if instance.iloc[id_feature - 1] >= threshold else -1
                elif operator == OperatorCondition.GT:
                    sign = 1 if instance.iloc[id_feature - 1] > threshold else -1
                elif operator == OperatorCondition.LE:
                    sign = 1 if instance.iloc[id_feature - 1] <= threshold else -1
                elif operator == OperatorCondition.LT:
                    sign = 1 if instance.iloc[id_feature - 1] < threshold else -1
                elif operator == OperatorCondition.EQ:
                    sign = 1 if instance.iloc[id_feature - 1] == threshold else -1
                elif operator == OperatorCondition.NEQ:
                    sign = 1 if instance.iloc[id_feature - 1] != threshold else -1
                else:
                    raise NotImplementedError("The operator " + str(self.operator) + " is not implemented.")

                output.append(sign * self.map_features_to_id_binaries[key][0])  # map_features_to_id_binaries[key][0] => id_binary
            return CNFencoding.format(output)
        assert True, "To implement"
        return output


    def get_id_features(self, binary_representation):
        return tuple(self.map_id_binaries_to_features[abs(lit)][0] for lit in binary_representation)


    # Return a map 'dict_id_features_to_id_binaries': : dict[id_feature] -> [list of id_binaries]
    def get_id_binaries(self):
        dict_id_features_to_id_binaries = dict()
        for key in self.map_features_to_id_binaries.keys():
            id_feature, _, _ = key
            id_binary = self.map_features_to_id_binaries[key][0]
            if id_feature in dict_id_features_to_id_binaries.keys():
                dict_id_features_to_id_binaries[id_feature].append(id_binary)
            else:
                dict_id_features_to_id_binaries[id_feature] = [id_binary]
        return dict_id_features_to_id_binaries


    def convert_features_to_dict_features(self, features, feature_names):
        dict_features = dict()
        dict_features_categorical = dict()

        for feature in features:
            name = feature["name"]
            if "theory" in feature.keys():
                theory = feature["theory"]
            else:
                theory = None
            if theory is not None and theory[0] == "categorical":
                dict_features_categorical[name] = theory[1][0]
                name = theory[1][0]
            
            if name not in dict_features.keys():
                dict_features[name] = [feature]
            else:
                dict_features[name].append(feature)

        # Sort the dict in order to have the features in the good order
        order_dict_features = collections.OrderedDict()
        for name in feature_names: # TODO Add warning when skipping a feature or if a name is present many times
            if name in dict_features.keys():
                order_dict_features[name] = dict_features[name]
            elif name in dict_features_categorical.keys():
                if dict_features_categorical[name] not in order_dict_features.keys(): # Add a warning, what is this exactly checking ?
                    order_dict_features[dict_features_categorical[name]] = dict_features[dict_features_categorical[name]]
        return order_dict_features


    def _apply_sign_on_operator(self, operator:OperatorCondition, sign:bool) -> OperatorCondition:
        """
        Invert the operator if there is a sign (a negative literal).
        """
        if not isinstance(sign, bool):
            raise ValueError('sign must be boolean')
        if sign is False: return operator

        if operator == OperatorCondition.GE:
            return OperatorCondition.LT
        if operator == OperatorCondition.GT:
            return OperatorCondition.LE
        if operator == OperatorCondition.LE:
            return OperatorCondition.GT
        if operator == OperatorCondition.LT:
            return OperatorCondition.GE
        if operator == OperatorCondition.EQ:
            return OperatorCondition.NEQ
        if operator == OperatorCondition.NEQ:
            return OperatorCondition.EQ

        raise NotImplementedError("The operator " + str(operator) + " is not implemented.")


    def to_features(self, reason,
                    eliminate_redundant_features:bool=True,
                    details:bool=False,
                    contrastive:bool=False,
                    without_intervals:bool=False,
                    feature_names=None) -> list[str] | dict: # TO DO feature_names can't be None here ! orders of arguments must be modified
        """
        Convert an implicant into features. 
            reason : implicant to convert
            eliminate_redundant_features : Define if a feature weaker than another one should be kept as a>2 when there is already a>3
            details : If True return is a dict containing ['id', 'name', 'threshold', 'operator', 'weight', 'string'] + theory if available
                            "id": The id of the feature (from 1 to the number of features) of the condition node represented by the literal of the implicant.
                            "name": The name of the feature. An alternative name is used if features_name are unknown
                            "threshold": The threshold of the condition node represented by the literal of the implicant.
                            "operator": The comparision to apply with the threshold.
                            "weight": Weight given to the feature (optional)
                            "string" : string representation
                      If False return a list of str reprensenting each feature
            contrastive : Is the reason contrastive
            without_intervals : Intervals representation as 2 inequalities (False) or as 1 interval (True), with details=True this impact the "string" key of the dict
            feature_names : Corresponding features names to consider

        Remark: call self.eliminate_redundant_features()
        """
        if eliminate_redundant_features:
            reason = self.eliminate_redundant_features(reason, contrastive)

        result = []        
        # First pass on each literal
        for lit in reason:
            feature_id, operator = self.map_id_binaries_to_features[abs(lit)][:2]
            sign = False if lit > 0 else True
            feature = {
                "id" : feature_id,
                "name" : feature_names[feature_id - 1],
                #"operator" : self.map_id_binaries_to_features[abs(lit)][1], # Useless ? 04/12/2025 # TO DO Remove old code
                #"sign" : sign,  # True if there is a sign else False
                "operator_sign_considered" : self._apply_sign_on_operator(operator, sign), # TO DO rename into operator
                "threshold" : self.map_id_binaries_to_features[abs(lit)][2],
                "weight" : reason[lit] if isinstance(reason, dict) else None }

            if len(self.map_check_already_used) != 0: 
                if self.map_check_already_used[feature["id"]] == "binary":
                    feature["theory"] = ("binary", (0,1))
                elif self.map_check_already_used[feature["id"]] == "categorical":
                    feature["theory"] = ("categorical", self.values_categorical_features[feature["name"]])
                elif self.map_check_already_used[feature["id"]] == "numerical":
                    feature["theory"] = ("numerical", )
            
            result.append(feature)

        # Get conditions with the same feature and create the string representation
        dict_features = self.convert_features_to_dict_features(result, feature_names)
        
        simple_result = []
        if without_intervals is True or eliminate_redundant_features is False:
            for name in dict_features.keys():
                features = dict_features[name]
                for feature in features:
                    str_operator = feature["operator_sign_considered"].to_str_readable()
                    feature["string"] = str(feature["name"]) + " " + str_operator + " " + str(feature["threshold"])
                    simple_result.append(feature["string"])
            if details is True:
                return dict_features
            return simple_result

        for name in dict_features.keys():
            features = dict_features[name]
            if "theory" in features[0].keys() and features[0]["theory"][0] == "binary":
                # Binary case 
                if len(features) != 1:
                    raise ValueError("A binary feature must be represented only by one condition in the explanation.")
                feature = features[0]
                
                operator = feature["operator_sign_considered"]
                if (operator == OperatorCondition.EQ or operator == OperatorCondition.NEQ):
                    #if feature["threshold"] != 0 and feature["threshold"] != 1:  # 04/12/2025 # TO DO Remove old code
                    #    raise ValueError("The threshold of a binary feature with EQ or NEQ must be equal to 0 or 1: " + str(feature["threshold"]))
                    feature["string"] = str(feature["name"]) + " " + operator.to_str_readable() + " " + str(feature["threshold"])
                elif (operator == OperatorCondition.GE or operator == OperatorCondition.GT):
                    #if feature["threshold"] != 0.5:  # 04/12/2025 # TO DO Remove old code
                    #    raise ValueError("The threshold of a binary feature with GE, GT, LE or LT must be equal to 0.5: " + str(feature["threshold"]))
                    feature["string"] = str(feature["name"]) + " = 1"
                elif (operator == OperatorCondition.LE or operator == OperatorCondition.LT):
                    #if feature["threshold"] != 0.5:  # 04/12/2025 # TO DO Remove old code
                    #    raise ValueError("The threshold of a binary feature with GE, GT, LE or LT must be equal to 0.5: " + str(feature["threshold"]))
                    feature["string"] = str(feature["name"]) + " = 0"
                else:
                    raise ValueError("The operator " + str(operator) + " do not exist.")
                    feature["string"] = str(feature["name"]) + " = 0"
                simple_result.append(feature["string"])    

            elif "theory" in features[0].keys() and features[0]["theory"][0] == "categorical":
                # Categorical case 
                thresholds = [feature["threshold"] for feature in features]
                sign_is_EQ = None
                if all(feature["operator_sign_considered"] == OperatorCondition.GE \
                        or feature["operator_sign_considered"] == OperatorCondition.GT \
                        or feature["operator_sign_considered"] == OperatorCondition.LE \
                        or feature["operator_sign_considered"] == OperatorCondition.LT
                        for feature in features):
                    sign_is_EQ = False
                elif all(feature["operator_sign_considered"] == OperatorCondition.EQ \
                        or feature["operator_sign_considered"] == OperatorCondition.NEQ \
                        for feature in features):
                    sign_is_EQ = True
                    if not all(feature["threshold"] == 0 or feature["threshold"] == 1 for feature in features):
                        raise ValueError("The thresholds of a categorical feature with EQ or NEQ must be equal to 0 or 1: " + str(feature["threshold"]))
                else:
                    raise ValueError("A categorical feature have some different operators: "+str(list(feature["operator_sign_considered"] for feature in features)))
                
                positive_values = []
                negative_values = []
                if sign_is_EQ is False:
                    for feature in features:
                        if feature["operator_sign_considered"].to_str_readable() == ">" or feature["operator_sign_considered"].to_str_readable() == ">=":
                            positive_values.append(feature)
                        else:
                            negative_values.append(feature)
                elif sign_is_EQ is True:
                    for feature in features:
                        if feature["operator_sign_considered"].to_str_readable() == "==":
                            positive_values.append(feature)
                        else:
                            negative_values.append(feature)
                else: # Theory is not in keys
                    raise ValueError("A categorical feature have bad operators: "+str(list(feature["operator_sign_considered"] for feature in features)))
                  
                #if len(positive_values) != 0 and len(negative_values) != 0:  # 04/12/2025 # TO DO Remove old code
                #    raise ValueError("Theory prevents to have at the same time a categorical equal to A and not equal to B.")
                if len(positive_values) > 1:
                    raise ValueError("Theory prevents to have at the same time a categorical equal to A and also equal to B.")
                
                if len(negative_values) != 0 and len(positive_values) != 0:
                    # Both
                    if contrastive is True:
                        name = negative_values[0]["theory"][1][0]
                        values = [feature["theory"][1][1] for feature in negative_values]
                        for feature in negative_values:
                            if len(values) == 1:
                                str_ = str(name) + " != " + str(values[0])
                            else:
                                str_ = str(name) + " != {" + ",".join(str(value) for value in values)+"}"
                            feature["string"] = str_
                        simple_result.append(feature["string"])

                        feature = positive_values[0]
                        feature["string"] = str_
                    else:
                        feature = positive_values[0]
                        name = feature["theory"][1][0]
                        value = feature["theory"][1][1]
                        str_ = str(name) + " = " + str(value)
                        feature["string"] = str_
                        simple_result.append(feature["string"])
                        for feature in negative_values:
                            feature["string"] = str_
                elif len(negative_values) != 0:
                    # Case color != {'green', 'red'}
                    name = negative_values[0]["theory"][1][0]
                    values = [feature["theory"][1][1] for feature in negative_values]
                    for feature in negative_values:
                        if len(values) == 1:
                            feature["string"] = str(name) + " != " + str(values[0])
                        else:
                            feature["string"] = str(name) + " != {" + ",".join(str(value) for value in values)+"}"
                    simple_result.append(feature["string"])

                elif len(positive_values) != 0:
                    # Case color = 'green' 
                    feature = positive_values[0]
                    name = feature["theory"][1][0]
                    value = feature["theory"][1][1]
                    feature["string"] = str(name) + " = " + str(value)
                    simple_result.append(feature["string"])
                
                    
            elif len(features) == 1:
                feature = features[0]
                str_operator = feature["operator_sign_considered"].to_str_readable()
                feature["string"] = str(feature["name"]) + " " + str_operator + " " + str(feature["threshold"])
                simple_result.append(feature["string"])
            elif len(features) == 2:
                feature_1 = features[0]
                feature_2 = features[1]
                operator_1 = feature_1["operator_sign_considered"]
                operator_2 = feature_2["operator_sign_considered"]

                if (operator_1 == OperatorCondition.GE or operator_1 == OperatorCondition.GT) \
                        and (operator_2 == OperatorCondition.GE or operator_2 == OperatorCondition.GT):
                    value = max(feature_1["threshold"], feature_2["threshold"])
                    operator = operator_1 if value == feature_1["threshold"] else operator_2
                    str_operator = operator.to_str_readable()
                    feature_1["string"] = str(feature_1["name"]) + " " + str_operator + " " + str(value)
                    feature_2["string"] = feature_1["string"]
                    simple_result.append(feature_1["string"])
                elif (operator_1 == OperatorCondition.LE or operator_1 == OperatorCondition.LT) \
                        and (operator_2 == OperatorCondition.LE or operator_2 == OperatorCondition.LT):
                    value = min(feature_1["threshold"], feature_2["threshold"])
                    operator = operator_1 if value == feature_1["threshold"] else operator_2
                    str_operator = operator.to_str_readable()
                    feature_1["string"] = str(feature_1["name"]) + " " + str_operator + " " + str(value)
                    feature_2["string"] = feature_1["string"]
                    simple_result.append(feature_1["string"])
                elif ((operator_1 == OperatorCondition.GE or operator_1 == OperatorCondition.GT) \
                      and (operator_2 == OperatorCondition.LE or operator_2 == OperatorCondition.LT)) or \
                        ((operator_1 == OperatorCondition.LE or operator_1 == OperatorCondition.LT) \
                         and (operator_2 == OperatorCondition.GE or operator_2 == OperatorCondition.GT)):

                    if feature_1["threshold"] > feature_2["threshold"]:
                        if operator_1 == OperatorCondition.LE or operator_1 == OperatorCondition.LT:
                            # case 1: [bound_1, bound_2]
                            bound_1 = feature_2["threshold"]
                            bound_2 = feature_1["threshold"]
                            bracket_1 = "[" if operator_2 == OperatorCondition.GE else "]"
                            bracket_2 = "]" if operator_1 == OperatorCondition.LE else "["
                            feature_1["string"] = str(feature_1["name"]) + " in " + bracket_1 + str(bound_1) + ", " + str(bound_2) + bracket_2
                            feature_2["string"] = feature_1["string"]
                            simple_result.append(feature_1["string"])
                        else:
                            # case 2: [infinity, bound_1] and [bound_2, infinity]
                            bound_1 = feature_2["threshold"]
                            bound_2 = feature_1["threshold"]
                            bracket_1 = "]" if operator_2 == OperatorCondition.LE else "["
                            bracket_2 = "[" if operator_1 == OperatorCondition.GE else "]"
                            interval_1 = "[infinity, " + str(bound_1) + bracket_1
                            interval_2 = bracket_2 + str(bound_2) + "infinity]"

                            feature_1["string"] = str(feature_1["name"]) + " in " + interval_1 + " and " + interval_2
                            feature_2["string"] = feature_1["string"]
                            simple_result.append(feature_1["string"])
                    else:
                        # feature_1["threshold"] <= feature_2["threshold"]:
                        if operator_1 == OperatorCondition.LE or operator_1 == OperatorCondition.LT:
                            # case 2: [infinity, bound_1] and [bound_2, infinity]
                            bound_1 = feature_1["threshold"]
                            bound_2 = feature_2["threshold"]
                            bracket_1 = "]" if operator_1 == OperatorCondition.LE else "["
                            bracket_2 = "[" if operator_2 == OperatorCondition.GE else "]"
                            interval_1 = "[infinity, " + str(bound_1) + bracket_1
                            interval_2 = bracket_2 + str(bound_2) + "infinity]"

                            feature_1["string"] = str(feature_1["name"]) + " in " + interval_1 + " and " + interval_2
                            feature_2["string"] = feature_1["string"]
                            simple_result.append(feature_1["string"])
                        else:
                            # case 1: [bound_1, bound_2]
                            bound_1 = feature_1["threshold"]
                            bound_2 = feature_2["threshold"]
                            bracket_1 = "[" if operator_1 == OperatorCondition.GE else "]"
                            bracket_2 = "]" if operator_2 == OperatorCondition.LE else "["
                            feature_1["string"] = str(feature_1["name"]) + " in " + bracket_1 + str(bound_1) + ", " + str(bound_2) + bracket_2
                            feature_2["string"] = feature_1["string"]
                            simple_result.append(feature_1["string"])

                else:
                    raise NotImplementedError("Combine operators " + str(operator_1) + " and " + str(operator_2) + " is not implemented.")
            else:
                # TO DO case to implement len(features) > 2:
                raise ValueError("In the interval view, the number of conditions for one feature must be <= 2.")

        if details is True:
            return dict_features
        return simple_result


    def extract_excluded_features(self, implicant, excluded_features):
        """Return index of the implicant to exclude.

        Args:
            implicant (_type_): _description_
            excluded_features (_type_): _description_

        Returns:
            :obj:`list` of int: List of index of the implicant to exclude
        """
        return [i for i, feature in enumerate(self.to_features(implicant, eliminate_redundant_features=False, details=True)) if
                feature["id_feature"] in excluded_features]


    def add_redundant_features(self, dict_redondant, literal, id_feature, operator, threshold, weight=None):
        if id_feature not in dict_redondant:  # add in the dict a new key
            dict_redondant[id_feature] = [(literal, threshold, operator, weight)]
        else:  # The key already exists
            self.n_redundant_features += 1
            dict_redondant[id_feature].append((literal, threshold, operator, weight))


    def eliminate_redundant_features(self, reason, contrastive=False):
        """
        A implicant without redundant features i.e. If we have 'feature_a > 3' and 'feature_a > 2', we keep only the id_binary linked to the boolean
        corresponding to the 'feature_a > 3' Warning, the 'implicant' parameter can be a list of literals or a dict of literals.
        In the last case, it is a map literal -> weight.
        Warning: reason can be either a list of literals or a dict literal -> weight
        """

        self.n_redundant_features = 0  # reset this information
        positive_literals_GE_GT = {}  # (id_feature) => [(threshold, operator, weight)]
        negative_literals_GE_GT = {}  # (id_feature) => [(threshold, operator, weight)]
        positive_literals_LE_LT = {}  # (id_feature) => [(threshold, operator, weight)]
        negative_literals_LE_LT = {}  # (id_feature) => [(threshold, operator, weight)]
        undone = []  # For undone literals in this elimination of redundant_features
        # For a contrastive, invert the sign and the reason, process to the elimination and re-inverse.
        if contrastive is True:
            reason = [-i for i in reason]

        # Search redundant features
        for literal in reason:
            key = self.map_id_binaries_to_features[abs(literal)]
            id_feature = key[0]
            operator = key[1]
            threshold = key[2]
            weight = reason[literal] if isinstance(reason, dict) else None
            if operator == OperatorCondition.GE or operator == OperatorCondition.GT:
                if literal > 0:  # If the sign is positive
                    self.add_redundant_features(positive_literals_GE_GT, literal, id_feature, operator, threshold, weight)
                else:  # If the sign is negative
                    self.add_redundant_features(negative_literals_GE_GT, literal, id_feature, operator, threshold, weight)
            elif operator == OperatorCondition.LE or operator == OperatorCondition.LT:
                if literal > 0:  # If the sign is negative
                    self.add_redundant_features(positive_literals_LE_LT, literal, id_feature, operator, threshold, weight)
                else:  # If the sign is positive
                    self.add_redundant_features(negative_literals_LE_LT, literal, id_feature, operator, threshold, weight)
            else:
                undone.append((literal, threshold, operator, weight))

        # Compute the condition to keep
        results = undone.copy()  # keep the undone

        for key in positive_literals_GE_GT.keys():
            max_positive_literals_GE_GT = positive_literals_GE_GT[key][argmax(tuple(x[1] for x in positive_literals_GE_GT[key]))]
            results.append(max_positive_literals_GE_GT)
        for key in negative_literals_GE_GT.keys():
            min_negative_literals_GE_GT = negative_literals_GE_GT[key][argmin(tuple(x[1] for x in negative_literals_GE_GT[key]))]
            results.append(min_negative_literals_GE_GT)

        for key in positive_literals_LE_LT.keys():
            min_positive_literals_LE_LT = positive_literals_LE_LT[key][argmin(tuple(x[1] for x in positive_literals_LE_LT[key]))]
            results.append(min_positive_literals_LE_LT)
        for key in negative_literals_LE_LT.keys():
            min_negative_literals_LE_LT = negative_literals_LE_LT[key][argmax(tuple(x[1] for x in negative_literals_LE_LT[key]))]
            results.append(min_negative_literals_LE_LT)

        # For a contrastive, re-inverse the sign.
        if contrastive is True:
            results = [(-result[0], result[1], result[2], result[3]) for result in results]

        # Return the good results according to the type of the reason (dict or list)
        if not isinstance(reason, dict):
            return tuple(result[0] for result in results)
        return {result[0]: result[3] for result in results}
