import random
import json
from typing import Iterable

from pyxai.sources.core.explainer.Visualisation import Visualisation
from pyxai.sources.core.tools.utils import count_dimensions
from pyxai.sources.core.structure.type import TypeFeature, OperatorCondition
from pyxai.sources.solvers.SAT.glucoseSolver import GlucoseSolver
from pyxai.sources.solvers.ENCORE.ENCORESolver import EncoreSolver
from pyxai import Tools


class Explainer:
    TIMEOUT = -1

    def __init__(self, do_history=True):
        self.target_prediction = None
        self._binary_representation = None
        self._elapsed_time = 0
        self._excluded_literals = []
        self._excluded_features = []
        self._instance = None
        self._theory = False
        self._categorical_features = []
        self._visualisation = Visualisation(self, do_history)
        self._glucose = None
        self._reference_instances = None
        self._last_features_types = None

    def get_model(self):
        """
        The model associated to the explainer (DT, RF, BT)
        """
        if hasattr(self, 'tree'):
            return self.tree
        elif hasattr(self, 'random_forest'):
            return self.random_forest
        elif hasattr(self, 'regression_boosted_trees'):
            return self.regression_boosted_trees
        elif hasattr(self, 'boosted_trees'):
            return self.boosted_trees

    @property
    def instance(self):
        """
        The instance to be explained.
        """
        return self._instance

    @property
    def binary_representation(self):
        """
        The binary representation of the selected instance.
        """
        return self._binary_representation

    @property
    def elapsed_time(self):
        """
        The time in second of the last call to an explanation. Equal to Explainer.TIMEOUT if time_limit was reached.
        """
        return self._elapsed_time

    def set_instance(self, instance):
        """
        Changes the instance on which explanations must be calculated.

        @param instance (list[float] | tuple[float]): the instance to be explained.
        """
        if instance is None:
            raise ValueError("The instance parameter is None.")

        if count_dimensions(instance) != 1:
            raise ValueError("The instance parameter should be an iterable of only one dimension (not " + str(
                count_dimensions(instance)) + ").")

        # The target observation.
        self._instance = instance
        # A binary representation of self.tree (a term that implies the tree)
        self._binary_representation = self._to_binary_representation(instance)

        # The target prediction (0 or 1)
        self.target_prediction = self.predict(self._instance)
        self.set_excluded_features(self._excluded_features)

    def count_features_before_converting(self, features):
        c = set()
        for feature in features:
            id = feature["id"]
            c.add(self.map_indexes[id])
        return len(c)

    def get_feature_names(self):
        """
        The name of features in the model
        """
        model = self.get_model()
        if model.learner_information is None or model.learner_information.feature_names is None:
            return ["f" + str(i + 1) for i in range(model.get_used_features())] + ["p"]
        return model.learner_information.feature_names

    def set_features_type(self, features_types):
        """
                Add a theory (related to the type of features) in the explainer.
                @param features_types (str | list): the theory selected.
        """
        self._last_features_types = features_types
        model = self.get_model()
        # To avoid a bug when several init done :)
        model.clear_theory_features()
        feature_names = self.get_feature_names()
        # reg_exp = regular expression 
        # "A*" is the regular expression meaning that ["A1", "A2", "A3"] come from one initial categorical feature that is one hot encoded 
        # to form 3 features. 

        self._reg_exp_categorical_features = dict()  # dict[reg_exp_feature]->[feature matching with A]

        self._numerical_features = []  # List of feature names of numerical_features
        self._categorical_features = []  # List of feature names of categorical features
        self._binary_features = []  # List of feature names of binary features
        self._values_categorical_features = dict()  # dict[feature name]->categorical value

        # Build the lists
        if isinstance(features_types, str):
            f = open(features_types)
            dict_types = json.load(f)
            f.close()
            for feature in dict_types.keys():
                if dict_types[feature]["type:"] != "Classification" and dict_types[feature]["type:"] != "Regression":
                    t = TypeFeature.from_str(dict_types[feature]["type:"])
                    encoder = dict_types[feature]["encoder:"]
                    if t == TypeFeature.CATEGORICAL:
                        if encoder != "OneHotEncoder":
                            raise NotImplementedError  # A voir avec Gilles
                        original_feature = dict_types[feature]["original_feature:"]
                        if original_feature in self._reg_exp_categorical_features:
                            self._reg_exp_categorical_features[original_feature].append(feature)
                        else:
                            self._reg_exp_categorical_features[original_feature] = [feature]
                        original_values = dict_types[feature]["original_values:"]
                        original_values = [original_feature, original_values[0], original_values[1]]
                        self._values_categorical_features[feature] = original_values
                        self._categorical_features.append(feature)
                    elif t == TypeFeature.BINARY:
                        self._binary_features.append(feature)
                    elif t == TypeFeature.NUMERICAL:
                        self._numerical_features.append(feature)

        elif isinstance(features_types, dict):

            default = None
            for key in features_types.keys():
                if key not in ["numerical", "categorical", "binary"]:
                    raise ValueError("The keys have to be either 'numerical' or 'categorical' or 'binary'.")

                if features_types[key] == TypeFeature.DEFAULT:
                    if default is not None:
                        raise ValueError("TypeFeature.DEFAULT must appear only once.")
                    default = key
                    continue
                elif key == "numerical":
                    for feature in features_types[key]:
                        if feature in feature_names:
                            self._numerical_features.extend([feature])
                        else:
                            raise ValueError("The feature " + str(feature) + " does not exist.")
                elif key == "binary":
                    for feature in features_types[key]:
                        if feature in feature_names:
                            self._binary_features.extend([feature])
                        else:
                            raise ValueError("The feature " + str(feature) + " does not exist.")
                elif key == "categorical":
                    if not isinstance(features_types[key], dict):
                        raise ValueError(
                            "The value of the key 'categorical' must be a Python dictionnary. Example: {'color': ('blue', 'red', 'yellow')}")
                    for feature in features_types[key].keys():
                        values = features_types[key][feature]
                        if "*" in feature:
                            feature = feature.split("*")[0]
                            associated_features = [f for f in feature_names if f.startswith(feature)]
                            if associated_features == []:
                                raise ValueError("No feature with the pattern " + str(feature) + ".")
                            if len(associated_features) != len(values):
                                raise ValueError("The number of values in " + str(
                                    values) + " must be equal to the number of feature with the pattern " + feature + " : " + str(
                                    associated_features))
                            for i in range(len(associated_features)):
                                self._values_categorical_features[associated_features[i]] = [feature, values[i], values]
                            self._reg_exp_categorical_features[feature] = associated_features
                            self._categorical_features.extend(associated_features)
                        elif "{" in feature and "}" in feature:
                            feature_1 = feature.split("{")[0]
                            feature_2 = feature.split("{")[1].split("}")[0].split(",")
                            associated_features = [feature_1 + feature_3 for feature_3 in feature_2]
                            if associated_features == []:
                                raise ValueError("No feature with the pattern " + str(feature) + ".")
                            if len(associated_features) != len(values):
                                raise ValueError("The number of values in " + str(
                                    values) + " must be equal to the number of feature with the pattern " + feature + " : " + str(
                                    associated_features))
                            for i in range(len(associated_features)):
                                self._values_categorical_features[associated_features[i]] = [feature, values[i], values]
                            self._reg_exp_categorical_features[feature] = associated_features
                            self._categorical_features.extend(associated_features)
                        else:
                            raise ValueError("Only one-hot encoded categorical features are take into account.")
                            if feature in feature_names:
                                self._reg_exp_categorical_features[feature] = [feature]
                                self._categorical_features.extend([feature])
                                for i in range(len(values)):
                                    self._values_categorical_features[feature + "_" + str(i)] = [feature, values[i],
                                                                                                 values]
                            else:
                                raise ValueError("The feature " + str(feature) + " does not exist.")
            if default is not None:
                # Without the last that is the label/prediction
                if default == "numerical":
                    self._numerical_features = [feature for feature in feature_names[:-1] if
                                                feature not in self._categorical_features and feature not in self._binary_features]
                elif default == "categorical":
                    self._categorical_features = [feature for feature in feature_names[:-1] if
                                                  feature not in self._numerical_features and feature not in self._binary_features]
                elif default == "binary":
                    self._binary_features = [feature for feature in feature_names[:-1] if
                                             feature not in self._numerical_features and feature not in self._categorical_features]
        else:
            raise ValueError(
                "The parameter must be either a list of string as ['Method*', 'CouncilArea*', 'Regionname*'] or a filename of .types file.")

        # Activate the theory
        self.activate_theory()

        self.map_indexes = dict()  # Used to count used_features_without_one_hot_encoded
        # Firstly, for the numerical one
        for feature in self._numerical_features:
            # Warning ids of features start from 1 to n (not 0), this is why there is +1 here.
            index = feature_names.index(feature) + 1
            model.add_numerical_feature(index)
            self.map_indexes[index] = index

        # Secondly, for the binary one
        for feature in self._binary_features:
            # Warning ids of features start from 1 to n (not 0), this is why there is +1 here.
            index = feature_names.index(feature) + 1
            model.add_binary_feature(index)
            self.map_indexes[index] = index

        # Finaly, for the categorical one
        for reg_exp_feature_name in self._reg_exp_categorical_features.keys():
            indexes = [feature_names.index(feature) + 1 for feature in
                       self._reg_exp_categorical_features[reg_exp_feature_name]]
            model.add_categorical_feature_one_hot(reg_exp_feature_name, indexes)
            for index in indexes:
                self.map_indexes[index] = reg_exp_feature_name

        model.set_values_categorical_features(self._values_categorical_features)
        # Display some statistics
        nNumerical = len(self._numerical_features)
        nBinaries = len(self._binary_features)
        nCategorical = len(self._reg_exp_categorical_features.keys())

        Tools.verbose("---------   Theory Feature Types   -----------")
        Tools.verbose("Before the one-hot encoding of categorical features:")
        Tools.verbose("Numerical features:", nNumerical)
        Tools.verbose("Categorical features:", nCategorical)
        Tools.verbose("Binary features:", nBinaries)
        Tools.verbose("Number of features:", nNumerical + nCategorical + nBinaries)
        Tools.verbose("Characteristics of categorical features:", self._values_categorical_features)
        used_features = set()
        used_features_without_one_hot_encoded = set()
        for key in model.map_features_to_id_binaries.keys():
            if key[0] not in self.map_indexes.keys():
                raise ValueError("The feature " + feature_names[key[0] - 1] + " is missing in features_types.")
            used_features.add(key[0])
            used_features_without_one_hot_encoded.add(self.map_indexes[key[0]])
        Tools.verbose("")
        Tools.verbose("Number of used features in the model (before the encoding of categorical features):",
                      len(used_features_without_one_hot_encoded))
        Tools.verbose("Number of used features in the model (after the encoding of categorical features):", len(used_features))
        Tools.verbose("----------------------------------------------")

    def _theory_clauses(self):
        raise NotImplementedError

    def activate_theory(self):
        """
        Add a theory in the resolution method.
        This allows to represent the fact that conditions depend on other conditions of a numerical attribute in explainer.
        """
        self._theory = True

    def deactivate_theory(self):
        """
        Unset the theory set with the activate_theory method.
        """
        self._theory = False

    def get_feature_names_from_literal(self, literal):
        """
        Given a literal in the binary representation, returns its feature name.
        Args:
            literal (int): the literal

        Returns: (str) the fname
        """
        dict_to_features = self.to_features([literal], eliminate_redundant_features=False, details=True)
        return dict_to_features[tuple(dict_to_features.keys())[0]][0]["name"]

    def set_excluded_features(self, excluded_features):
        """
        Sets the features that the user do not want to see in explanations. You must give the name of the features.

        @param excluded_features (list[str] | tuple[str]): the features name to be excluded
        """
        if len(excluded_features) == 0:
            self.unset_excluded_features()
            return
        self._excluded_features = excluded_features
        if self.instance is None:
            return
        bin_rep = self.extend_reason_to_complete_representation(
            []) if self._binary_representation is None else self._binary_representation
        self._excluded_literals = [lit for lit in bin_rep if
                                   self.get_feature_names_from_literal(lit) in excluded_features]

    def _set_specific_features(self, specific_features):  # TODO a changer en je veux ces features
        excluded = []
        if hasattr(self, 'tree'):
            excluded = [f for f in self.tree.learner_information.feature_names if f not in specific_features]
        elif hasattr(self, 'random_forest'):
            excluded = [f for f in self.random_forest.learner_information.feature_names if f not in specific_features]
        else:
            assert hasattr(self, 'boosted_tree')
            excluded = [f for f in self.boosted_tree.learner_information.feature_names if f not in specific_features]

        if excluded is None:
            raise NotImplementedError
        self.set_excluded_features(excluded)

    def unset_excluded_features(self):
        """
        Unset the features set with the set_excluded_features method.
        """
        self._excluded_literals = []
        self._excluded_features = []

    def _is_specific(self, lit):
        return lit not in self._excluded_literals

    def reason_contains_features(self, reason, features_name):
        """
        Returns True if the reason contains the feature_name.

        @param reason (list[int]): the reason
        @param features_name (str): the name of the feature
        @return: True if feature_name is part of the reason, False otherwise.
        """
        return any(self.get_feature_names_from_literal(lit) in features_name for lit in reason)

    def _to_binary_representation(self, instance):
        raise NotImplementedError

    def to_features(self, binary_representation, eliminate_redundant_features=True, details=False, contrastive=False,
                    without_intervals=False):
        """
        Converts a binary representation of a reason into the features space.
        
        When the parameter details is set to False, returns a Tuple of String where each String represents a condition
        “<id_feature> <operator> <threshold>” associated with the binary representation given as first parameter. By default, a string represents
        such a condition but if you need more information, you can set the parameter details to True. In this case, the method returns
        a Tuple of Dict where each dictionary provides more information on the condition. This method also allows one to eliminate redundant
        conditions. For example, if we have “feature_a > 3” and “feature_a > 2”, we keep only the binary variable linked to the Boolean corresponding
        to the “feature_a > 3”. Therefore, if you want to get all conditions, you have to set the parameter eliminate_redundant to False.

        @param binary_representation (list[int]): the binary representation to convert
        @param eliminate_redundant_features (bool): eliminate possible redundant features.
        @param details (bool): returns a detailed representation instead of a string
        @return: a list of string or of dictionaries depending the value of details.
        """
        raise NotImplementedError

    def sufficient_reason(self, *, n=1, seed=0):
        raise NotImplementedError

    def direct_reason(self):
        raise NotImplementedError

    def predict(self, instance):
        """
        return the prediction of the instance w.r.t. the classifier associated to the explainer.

        @param instance: (list[float]) the instance to be predicted
        @return: The prediction
        """
        raise NotImplementedError

    def is_implicant(self, binary_representation):
        """Check if the binary representation is an implicant of the instance"""
        raise NotImplementedError

    def extend_reason_with_theory(self, reason):
        """
        Given a partial binary representation, extend it in order to validate the theory.

        """
        if self._theory is False:
            return reason
        if self._glucose is None:
            self._glucose = GlucoseSolver()
            self._glucose.add_clauses(self.get_model().get_theory(self._binary_representation))
        return self._glucose.propagate(reason)[1]

    def is_reason(self, reason, *, n_samples=1000):
        """
        Return if the reason given in parameter is really a reason. Since the process can be time consuming, one limits the
        number of checks.

        @param reason: (list[float]) The reason to be tested.
        @param n_samples: (int) the number of tests to be done.
        @return: True if the reason is really one reason, False otherwise.
        """

        for _ in range(n_samples):
            binary_representation = self.extend_reason_to_complete_representation(reason)
            if not self.is_implicant(binary_representation):
                return False
        return True

    def is_sufficient_reason(self, reason, *, n_samples=50):
        """
        Checks if a reason is a sufficient one.

        This method checks wheter a reason is sufficient. To this purpose, we first call firstly the method is_reason to check whether n_samples
        complete binary representations from this reason (randomly generated) lead to the correct prediction or not. Then, we verify the minimality
        of the reason in the sense of set inclusion. To do that, we delete a literal of the reason, test with is_reason that this new binary
        representation is not a reason and put back this literal. The method repeats this operation on every literal of the reason.
        Because this method is based on random generation and a limited number of samples, it is not deterministic (i.e., it is not 100% sure
        to provide the right answer). Therefore, this method can return True, False or None.
        @param reason: (list[int]) the reason to be checked.
        @param n_samples: (int) the number of samples done.
        @return: True if it is a ssuficient reason( w.r.t. the number of samples), False if not and None if it is not sure.
        """
        if not self.is_reason(reason, n_samples=n_samples):
            return False  # We are sure it is not a reason
        tmp = list(reason)
        random.shuffle(tmp)
        i = 0
        for lit in tmp:
            copy_reason = list(reason).copy()
            copy_reason.remove(lit)
            copy_reason.append(-lit)
            if self.is_reason(copy_reason, n_samples=n_samples):
                return None
            i += 1
            if i > n_samples:
                break
        return True

    def is_contrastive_reason(self, reason):
        """
        Checks if a reason is a contrastive one.

        Checks if the reason is a contrastive one. Replaces in the binary representation of the instance each literal of the reason with its
        opposite and checks that the result does not predict the same class as the initial instance. Returns True if the reason is contrastive,
        False otherwise.
        @param reason: the reason to be checked.
        @return: True if it is a contrastive reason, False otherwise.
        """
        copy_binary_representation = list(self._binary_representation).copy()
        for lit in reason:
            copy_binary_representation[copy_binary_representation.index(lit)] = -lit
        return not self.is_implicant(copy_binary_representation)

    def is_contrastive_reason_instance(self, reason):
        """
        Checks if a reason is a contrastive one by changing the instance by the element of the reason.
        @instance: the instance to change 
        @param reason: the reason to be checked.
        @return: True if it is a contrastive reason, False otherwise.
        """
        features = self.to_features(reason, eliminate_redundant_features=False, details=True)
        copy_instance = list(self.instance).copy()
        for feature in features:
            id = feature["id"] - 1
            operator = feature["operator"]
            threshold = feature["threshold"]
            if operator == OperatorCondition.GE:
                # print("copy_instance[id]:", copy_instance[id])
                # assert copy_instance[id] < threshold, "We have to change that to apply the contrastive."
                copy_instance[id] = threshold
            else:
                raise NotImplementedError()
        new_binary_representation = self._to_binary_representation(copy_instance)
        return not self.is_implicant(new_binary_representation)

    def extend_reason_to_complete_representation(self, reason):
        complete = list(reason).copy()
        to_add = [literal for literal in self._binary_representation if
                  literal not in complete and -literal not in complete]
        complete = self.extend_reason_with_theory(complete)
        for literal in to_add:
            if literal in complete or -literal in complete:  # This line is too costly: TODO: Create a map
                continue
            sign = random.choice([1, -1])
            complete.append(sign * abs(literal))
            complete = self.extend_reason_with_theory(complete)
        assert len(complete) == len(self._binary_representation)
        return complete

    @staticmethod
    def format(reasons, n=1):
        if reasons is None or len(reasons) == 0:
            return tuple()
        if type(n) != int or n > 1:
            if isinstance(reasons[0], Iterable):
                return tuple(tuple(sorted(reason, key=lambda l: abs(l))) for reason in reasons)
            else:
                return tuple(tuple(sorted(reasons, key=lambda l: abs(l))))

        if isinstance(reasons[0], Iterable):
            return Explainer.format(reasons[0])
        return tuple(sorted(reasons, key=lambda l: abs(l)))

    def set_reference_instances(self, reference_instances):
        if not isinstance(reference_instances, dict):
            raise ValueError("The `reference_instances` parameter have to be a dict.")

        self._reference_instances = dict()
        for label in reference_instances.keys():
            binarized_instances = []
            for instance in reference_instances[label]:
                binarized_instances.append(self._to_binary_representation(instance))
            self._reference_instances[label] = tuple(binarized_instances)

    """_summary_

        Args:
            n_anchors (integer): Number of anchors to have in the explanation (example: a 2-anchored explanation)
            reference_instances (dictionary): Python dictionary where the keys are the labels and the values are lists contening instances.
            The instances of reference_instances are those for which the expert is sure of the classification (they are anchored).
    
        Returns:
            A n_anchors-anchored andutive explanation
        """

    def _most_anchored_reason(self, *, n_variables, cnf, time_limit=None, check=False, type_references="normal"):

        if self._reference_instances is None or not isinstance(self._reference_instances, dict):
            raise ValueError("Please use the set_reference_instances() before to use this method.")
        references = dict()

        if type_references == "normal":
            references = self._reference_instances
        elif type_references == "one_side":
            if self.target_prediction == 1:
                references[0] = []
                references[1] = self._reference_instances[1]

            else:
                references[0] = self._reference_instances[0]
                references[1] = []
        else:
            raise ValueError("Bad paramerter for 'type_references': " + str(type_references))

        print("size reference_instances 0: ", len(references[0]))
        print("size reference_instances 1: ", len(references[1]))

        solver = EncoreSolver(cnf, self.target_prediction, self._binary_representation, references, n_variables)

        n_anchors = 1
        go_next = True
        previous_reason = None
        time_used = 0
        while (go_next is True and (time_limit is None or time_used < time_limit)):
            # print("search k:", n_anchors)
            local_time_limit = None if time_limit is None else time_limit - time_used
            return_code, status, reason, time, last_k = solver.solve(n_anchors=n_anchors, time_limit=local_time_limit,
                                                                     with_check=check)
            time_used += time
            # define EXIT_NOT_FOUND 1    // rien trouvé!
            # define EXIT_UNSAT 10       // pas possible de trouver une explication ancrée ... try again :P
            # define EXIT_CANDIDATE 20   // j'ai trouvé une candidat, mais j'ai pas eu le temps de le réduire.
            # define EXIT_FOUND 30       // c'est bon, j'ai une explication ancré minimal.

            if reason is not None:
                previous_reason = reason
                go_next = True
                if last_k is not None:
                    n_anchors = int(last_k)
                self.last_n_anchors = n_anchors
                n_anchors += 1
            else:
                break

        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT

        if n_anchors == 1 and previous_reason is None:
            self.last_n_anchors = 0

        return None if previous_reason is None else Explainer.format(previous_reason)
    
    def simplify_theory(self, tree):
        if self._theory is True:
            solver = GlucoseSolver()
            theory_cnf = self.get_model().get_theory(None)
            return solver.symplify_theory(tree, theory_cnf)
        return tree
    
    @property
    def visualisation(self):
        """This object allows to open gui, save images, and so on
        see : https://www.cril.univ-artois.fr/pyxai/documentation/visualization/
        """
        return self._visualisation
