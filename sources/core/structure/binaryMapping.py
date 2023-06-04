from pyxai.sources.core.structure.type import OperatorCondition, TypeTheory
from pyxai.sources.core.tools.encoding import CNFencoding
from pyxai.sources.core.tools.GUI import GraphicalInterface

from numpy import argmax, argmin 
import collections

class BinaryMapping():

    def __init__(self, map_id_binaries_to_features, map_features_to_id_binaries, learner_information):
        self.map_id_binaries_to_features = map_id_binaries_to_features
        self.map_features_to_id_binaries = map_features_to_id_binaries
        self.map_numerical_features = {} # dict[id_feature] -> [id_binaries of the feature]
        self.map_binary_features = {} # dict[id_feature] -> [id_binaries of the feature]
        self.map_categorical_features_ordinal = {} # dict[id_feature] -> [id_binaries of the feature]
        self.map_categorical_features_one_hot = {} # dict[reg_exp_name] -> [id_binaries of the set of features representing the reg_exp_name of the feature that was one hot encoded]
        
        self.map_check_already_used = {} # Just to check if a feature is already used

        self.learner_information = learner_information

        self.n_redundant_features = 0 # Variable to store the number of redondances in eliminate_redundant_features()
        self.feature_names = None

    @property
    def raw_model(self):
        return self.learner_information.raw_model

    @property
    def accuracy(self):
        return self.learner_information.accuracy

    @property
    def features_name(self):
        return self.learner_information.feature_names

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
        
    def add_numerical_feature(self, id_feature):
        if id_feature in self.map_check_already_used.keys():
            raise ValueError("The feature id ("+str(id_feature)+") is already used as"+self.map_check_already_used[id_feature])
        
        id_binaries_of_the_feature = []
        for key in self.map_features_to_id_binaries.keys():
          if key[0] == id_feature:
            id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])

        self.map_numerical_features[id_feature] = id_binaries_of_the_feature
        self.map_check_already_used[id_feature] = "numerical"

    def add_binary_feature(self, id_feature):
        if id_feature in self.map_check_already_used.keys():
            raise ValueError("The feature id ("+str(id_feature)+") is already used as"+self.map_check_already_used[id_feature])
                
        id_binaries_of_the_feature = []
        for key in self.map_features_to_id_binaries.keys():
          if key[0] == id_feature:
            id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])

        self.map_binary_features[id_feature] = id_binaries_of_the_feature
        self.map_check_already_used[id_feature] = "binary"

    def add_categorical_feature_one_hot(self, reg_exp_feature_name, id_features): 
        id_binaries_of_the_feature = []
        for id_feature in id_features:
            if id_feature in self.map_check_already_used.keys():
                raise ValueError("The feature id ("+str(id_feature)+") is already used as "+self.map_check_already_used[id_feature])
        
        for key in self.map_features_to_id_binaries.keys():
            for id_feature in id_features:
                if key[0] == id_feature:
                    id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])
                
        self.map_categorical_features_one_hot[reg_exp_feature_name] = id_binaries_of_the_feature
        for id_feature in id_features:
            self.map_check_already_used[id_feature] = "categorical"


    def add_categorical_feature_ordinal(self, id_feature):
        raise NotImplementedError #See with Gilles ? on laisse pour l'instant
        if id_feature in self.map_numerical_features: 
            raise ValueError("The given id_feature (" + str(id_feature) + ") is already considered as numerical.")
        if id_feature in self.map_categorical_features_ordinal:
            raise ValueError("The given id_feature (" + str(id_feature) + ") is already considered as categorical ordinal.")
        
        id_binaries_of_the_feature = []
        for key in self.map_features_to_id_binaries.keys():
            if key[0] == id_feature:
                id_binaries_of_the_feature.append(self.map_features_to_id_binaries[key][0])

        self.map_categorical_features_ordinal[id_feature] = id_binaries_of_the_feature



    """
        
    """
    def get_theory(self, binary_representation, *, theory_type=TypeTheory.SIMPLE, id_new_var=0):
        #structure to help to do this method faster
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
            conditions = [tuple(list(self.map_id_binaries_to_features[id])+[id]) for id in id_binaries]
            conditions = sorted(conditions, key=lambda t: t[2], reverse=True)
            id_binaries_sorted = tuple(condition[3] for condition in conditions)
            
            for i in range(len(id_binaries_sorted)-1): #To not takes the last
                clauses.append((-id_binaries_sorted[i], id_binaries_sorted[i+1]))

            if theory_type == TypeTheory.NEW_VARIABLES: 
                id_new_var = id_new_var + 1
                #associated_literals = []
                for id_binary in id_binaries_sorted:
                    clauses.append((-id_new_var, map_id_binary_sign[id_binary]*id_binary))
                    map_is_represented_by_new_variables[id_binary] = True
                    # associated_literals.append(map_id_binary_sign[id_binary]*id_binary)
                new_variables.append(id_new_var)
    
        # For categorical features that was one hot encoded
        for key in self.map_categorical_features_one_hot.keys(): 
            id_binaries = self.map_categorical_features_one_hot[key]
            for i, id_1 in enumerate(id_binaries):
                for j, id_2 in enumerate(id_binaries):
                    if i != j:
                        # we code a => not b that is equivalent to not a or not b (material implication)
                        clauses.append((-id_1, -id_2))   

        # For binary feature, nothing to do.

        if theory_type == TypeTheory.SIMPLE:    
            return clauses
        elif theory_type == TypeTheory.NEW_VARIABLES:
            return clauses, (new_variables, map_is_represented_by_new_variables)
        else:
            raise NotImplementedError
        

    def compute_id_binaries(self):
        assert False, "Have to be implemented in a child class."


    def instance_to_binaries(self, instance, preference_order=None):
        """
        map_id_binaries_to_features: list[id_binary] -> (id_feature, operator, threshold)
        map_features_to_id_binaries: dict[(id_feature, operator, threshold)] -> [id_binary, n_appears, n_appears_per_tree]
        Transform a instance into a cube (conjunction of literals) according to the tree
        """
        output = []
        if preference_order is None:
            for key in self.map_features_to_id_binaries.keys():  # the keys are of the type: (id_feature, operator, threshold)
                id_feature = key[0]
                operator = key[1]
                threshold = key[2]

                if operator == OperatorCondition.GE:
                    sign = 1 if instance[id_feature - 1] >= threshold else -1
                elif operator == OperatorCondition.GT:
                    sign = 1 if instance[id_feature - 1] > threshold else -1
                elif operator == OperatorCondition.LE:
                    sign = 1 if instance[id_feature - 1] <= threshold else -1
                elif operator == OperatorCondition.LT:
                    sign = 1 if instance[id_feature - 1] < threshold else -1
                elif operator == OperatorCondition.EQ:
                    sign = 1 if instance[id_feature - 1] == threshold else -1
                elif operator == OperatorCondition.NEQ:
                    sign = 1 if instance[id_feature - 1] != threshold else -1
                else:
                    raise NotImplementedError("The operator " + str(self.operator) + " is not implemented.")

                output.append(sign * self.map_features_to_id_binaries[key][0])  # map_features_to_id_binaries[key][0] => id_binary
            return CNFencoding.format(output)
        assert True, "To implement"
        return output


    def get_id_features(self, binary_representation):
        return tuple(self.map_id_binaries_to_features[abs(lit)][0] for lit in binary_representation)


    def to_features(self, reason, eliminate_redundant_features=True, details=False, contrastive=False, graphical_interface=False):
        """
        Convert an implicant into features. Return a tuple of features.
        Two types of features are available according to the details parameter.
        If details is False, each feature is represented by a string "name >= threshold" or "name < threshold".
        Else, if details is True, each feature is a python dict with the keys "id", "name", "threshold", "sign" and "weight".
        "id": The id of the feature (from 1 to the number of features) of the condition node represented by the literal of the implicant.
        "name": The name of the feature. It is None if it is not available.
        "threshold": The threshold of the condition node represented by the literal of the implicant.
        "sign": If the associated literal was positive in the implicant (resp. negative), it is True (resp. False).
        "weight": To give a weight to the feature (optional)

        Remark: call self.eliminate_redundant_features()
        """
        result = []
        if eliminate_redundant_features:
            reason = self.eliminate_redundant_features(reason, contrastive)
        used_features = set()
        for key in self.map_features_to_id_binaries.keys():
            used_features.add(key[0])
        print("features:", used_features)
        for lit in reason:
            feature = dict()
            feature["id"] = self.map_id_binaries_to_features[abs(lit)][0]

            if self.learner_information is None:
                feature["name"] = "f" + str(feature["id"])
            elif self.learner_information.feature_names is None:
                feature["name"] = "f" + str(feature["id"])
            else:
                feature["name"] = self.learner_information.feature_names[feature["id"] - 1]
            
            feature["operator"] = self.map_id_binaries_to_features[abs(lit)][1]
            feature["threshold"] = self.map_id_binaries_to_features[abs(lit)][2]
            feature["sign"] = True if lit > 0 else False
            feature["weight"] = reason[lit] if isinstance(reason, dict) else None
            if details:
                result.append(feature)
            else:
                if feature["operator"] == OperatorCondition.GE:
                    str_sign = " >= " if feature["sign"] else " < "
                elif feature["operator"] == OperatorCondition.GT:
                    str_sign = " > " if feature["sign"] else " <= "
                elif feature["operator"] == OperatorCondition.LE:
                    str_sign = " <= " if feature["sign"] else " > "
                elif feature["operator"] == OperatorCondition.LT:
                    str_sign = " < " if feature["sign"] else " >= "
                elif feature["operator"] == OperatorCondition.EQ:
                    str_sign = " == " if feature["sign"] else " != "
                elif feature["operator"] == OperatorCondition.NEQ:
                    str_sign = " != " if feature["sign"] else " == "
                else:
                    raise NotImplementedError("The operator " + str(feature["operator"]) + " is not implemented.")

                result.append(str(feature["name"]) + str_sign + str(feature["threshold"]))
        if graphical_interface is True:
            gi = GraphicalInterface()
            gi.mainloop()
            
        return tuple(result)


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
        if id_feature not in dict_redondant: #add in the dict a new key
            dict_redondant[id_feature] = [(literal, threshold, operator, weight)]
        else: #The key already exists
            self.n_redundant_features += 1
            dict_redondant[id_feature].append((literal, threshold, operator, weight))
            

    def eliminate_redundant_features(self, reason, contrastive=False):
        """
        A implicant without redundant features i.e. If we have 'feature_a > 3' and 'feature_a > 2', we keep only the id_binary linked to the boolean
        corresponding to the 'feature_a > 3' Warning, the 'implicant' parameter can be a list of literals or a dict of literals.
        In the last case, it is a map literal -> weight.
        Warning: reason can be either a list of literals or a dict literal -> weight
        """

        self.n_redundant_features = 0 # reset this information
        positive_literals_GE_GT = {} # (id_feature) => [(threshold, operator, weight)]
        negative_literals_GE_GT = {} # (id_feature) => [(threshold, operator, weight)]
        positive_literals_LE_LT = {} # (id_feature) => [(threshold, operator, weight)]
        negative_literals_LE_LT = {} # (id_feature) => [(threshold, operator, weight)]
        undone = [] # For undone literals in this elimination of redundant_features
        #For a contrastive, invert the sign and the reason, process to the elimination and re-inverse. 
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
                if literal > 0: #If the sign is positive
                    self.add_redundant_features(positive_literals_GE_GT, literal, id_feature, operator, threshold, weight)
                else: #If the sign is negative
                    self.add_redundant_features(negative_literals_GE_GT, literal, id_feature, operator, threshold, weight)
            elif operator == OperatorCondition.LE or operator == OperatorCondition.LT:
                if literal > 0: #If the sign is negative 
                    self.add_redundant_features(positive_literals_LE_LT, literal, id_feature, operator, threshold, weight)
                else: #If the sign is positive
                    self.add_redundant_features(negative_literals_LE_LT, literal, id_feature, operator, threshold, weight)
            else:
                undone.append((literal, threshold, operator, weight))

        # Compute the condition to keep
        results = undone.copy() #keep the undone

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
        
        #For a contrastive, re-inverse the sign. 
        if contrastive is True:
            results = [(-result[0],result[1],result[2],result[3]) for result in results]
        
        # Return the good results according to the type of the reason (dict or list)
        if not isinstance(reason, dict):
            return tuple(result[0] for result in results)
        return {result[0]: result[3] for result in results}
        
