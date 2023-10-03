import random
import time

import c_explainer
import numpy

from pyxai.sources.core.explainer.Explainer import Explainer
from pyxai.sources.core.structure.type import Encoding, PreferredReasonMethod, TypeTheory, ReasonExpressivity
from pyxai.sources.core.tools.encoding import CNFencoding
from pyxai.sources.core.tools.utils import compute_weight
from pyxai.sources.solvers.MAXSAT.OPENWBOSolver import OPENWBOSolver
from pyxai.sources.solvers.MUS.MUSERSolver import MUSERSolver
from pyxai.sources.solvers.MUS.OPTUXSolver import OPTUXSolver
from pyxai.sources.solvers.SAT.glucoseSolver import GlucoseSolver


class ExplainerRF(Explainer):

    def __init__(self, random_forest, instance=None):
        """Create object dedicated to finding explanations from a random forest ``random_forest`` and an instance ``instance``.

        Args:
            random_forest (RandomForest): The model in the form of a RandomForest object.
            instance (:obj:`list` of :obj:`int`, optional): The instance (an observation) on which explanations must be calculated. Defaults to None.

        Raises:
            NotImplementedError: Currently, the explanations from a random forest are not available in the multi-class scenario (work in progress).
        """
        super().__init__()

        self._random_forest = random_forest
        self.c_RF = None

        if instance is not None:
            self.set_instance(instance)


    @property
    def random_forest(self):
        return self._random_forest


    def to_features(self, binary_representation, *, eliminate_redundant_features=True, details=False, contrastive=False, without_intervals=False):
        """
        Convert each literal of the implicant (representing a condition) to a tuple (``id_feature``, ``threshold``, ``sign``, ``weight``).
          - ``id_feature``: the feature identifier.
          - ``threshold``: threshold in the condition id_feature < threshold ?
          - ``sign``: indicate if the condition (id_feature < threshold) is True or False in the reason.
          - ``weight``: possible weight useful for some methods (can be None).

        Remark: Eliminate the redundant features (example: feature5 < 0.5 and feature5 < 0.4 => feature5 < 0.4).

        Args:
            binary_representation (obj:`list` of :obj:`int`): The reason or an implicant.

        Returns:
            obj:`tuple` of :obj:`tuple` of size 4: Represent the reason in the form of features (with their respective thresholds, signs and possible
            weights)
        """
        return self._random_forest.to_features(binary_representation, eliminate_redundant_features=eliminate_redundant_features, details=details,
                                               contrastive=contrastive, without_intervals=without_intervals, feature_names=self.get_feature_names())


    def _to_binary_representation(self, instance):
        return self._random_forest.instance_to_binaries(instance)


    def is_implicant(self, binary_representation):
        return self._random_forest.is_implicant(binary_representation, self.target_prediction)


    def predict(self, instance):
        return self._random_forest.predict_instance(instance)


    def direct_reason(self):
        """The direct reason of an instance x is the term t of the implicant (binary form of the instance) corresponding to the unique root-to-leaf
        path of the tree that covers x. (see the Trading Complexity for Sparsity
        in Random Forest Explanations paper (Gilles Audemard, Steve Bellart, Louenas Bounia, Frederic Koriche,
        Jean-Marie Lagniez and Pierre Marquis1) for more information)

        Returns:
            (obj:`list` of :obj:`int`): Reason in the form of literals (binary form). The to_features() method allows to obtain the features
            of this reason.
        """
        if self._instance is None:
            raise ValueError("Instance is not set")

        self._elapsed_time = 0
        tmp = [False for _ in range(len(self.binary_representation) + 1)]
        for tree in self._random_forest.forest:
            local_target_prediction = tree.predict_instance(self._instance)
            if local_target_prediction == self.target_prediction or self._random_forest.n_classes > 2:
                local_direct = tree.direct_reason(self._instance)
                for l in local_direct:
                    tmp[abs(l)] = True
        direct_reason = []
        for l in self.binary_representation:
            if tmp[abs(l)] :
                direct_reason.append(l)

        # remove excluded features
        if any(not self._is_specific(lit) for lit in direct_reason):
            reason = None
        else:
            reason = Explainer.format(list(direct_reason))
        self.add_history(self._instance, self.__class__.__name__, self.direct_reason.__name__, reason)
        return reason


    def minimal_contrastive_reason(self, *, n=1, time_limit=None):
        """Formally, a contrastive explanation for an instance x given f is a subset t of the characteristics of x that is minimal w.r.t. set
        inclusion among those such that at least one other instance x' that coincides with x except on the characteristics from t is not classified
        by f as x is. See the On the Explanatory Power of Decision Trees paper (Gilles Audemard, Steve Bellart, Louenas Bounia, Frédéric Koriche,
        Jean-Marie Lagniez, and Pierre Marquis) for more details. This method compute one or several minimal (in size) contrastive reasons.

        Args:
            n (int, optional): The desired number of reasons. Defaults to 1.

        Returns:
            (:obj:`tuple` of :obj:`tuple` of `int`): Reasons in the form of literals (binary form). The to_features() method allows to obtain
            the features of reasons. When only one reason is requested through the 'n' parameter, return just one :obj:`tuple` of `int`
            (and not a :obj:`tuple` of :obj:`tuple`).
        """
        if self._instance is None:
            raise ValueError("Instance is not set")

        n = n if type(n) == int else float('inf')
        first_call = True
        time_limit = 0 if time_limit is None else time_limit
        best_score = 0
        tree_cnf = self._random_forest.to_CNF(self._instance, self._binary_representation, target_prediction=1 if self.target_prediction == 0 else 0,
                                              tree_encoding=Encoding.SIMPLE)

        # structure to help to do this method faster
        map_in_binary_representation = dict()
        for id in self._binary_representation:
            map_in_binary_representation[id] = True
            map_in_binary_representation[-id] = False

        # print("tree_cnf:", tree_cnf)
        max_id_binary_representation = CNFencoding.compute_max_id_variable(self._binary_representation)
        # print("max_id_binary_representation:", max_id_binary_representation)

        max_id_binary_cnf = CNFencoding.compute_max_id_variable(tree_cnf)
        # print("max_id_variable:", max_id_binary_cnf)

        MAXSATsolver = OPENWBOSolver()
        # print("Length of the binary representation:", len(self._binary_representation))
        # print("Number of hard clauses in the CNF encoding the random forest:", len(tree_cnf))
        MAXSATsolver.add_hard_clauses(tree_cnf)
        if self._theory is False:
            for lit in self._binary_representation:
                MAXSATsolver.add_soft_clause([lit], weight=1)
        else:
            # Hard clauses
            theory_cnf, theory_new_variables = self._random_forest.get_theory(
                self._binary_representation,
                theory_type=TypeTheory.NEW_VARIABLES,
                id_new_var=max_id_binary_cnf)
            theory_new_variables, map_is_represented_by_new_variables = theory_new_variables
            # print("Number of hard clauses in the theory:", len(theory_cnf))
            MAXSATsolver.add_hard_clauses(theory_cnf)
            count = 0
            for lit in self._binary_representation:
                if map_is_represented_by_new_variables[abs(lit)] is False:
                    MAXSATsolver.add_soft_clause([lit], weight=1)
                    count += 1
            for new_variable in theory_new_variables:
                MAXSATsolver.add_soft_clause([new_variable], weight=1)

        # Remove excluded features
        for lit in self._excluded_literals:
            MAXSATsolver.add_hard_clause([lit])

        time_used = 0
        results = []
        while True:
            status, reason, _time = MAXSATsolver.solve(time_limit=time_limit)
            time_used += _time
            if time_limit != 0 and time_used > time_limit:
                break
            if reason is None:
                break
            # We have to invert the reason :)
            true_reason = [-lit for lit in reason if abs(lit) <= len(self._binary_representation) and map_in_binary_representation[-lit] == True]

            # Add a blocking clause to avoid this reason in the next steps
            MAXSATsolver.add_hard_clause([-lit for lit in reason if abs(lit) <= max_id_binary_representation])

            # Compute the score
            score = len(true_reason)
            # Stop or not due to score :)
            if first_call:
                best_score = score
            elif score != best_score:
                break
            first_call = False

            # Add this contrastive
            results.append(true_reason)

            # Stop or not due to time or n :)
            if (time_limit != 0 and time_used > time_limit) or len(results) == n:
                # print("End by time_limit or 'n' reached.")
                break

        self._elapsed_time = time_used if time_limit == 0 or time_used < time_limit else Explainer.TIMEOUT

        reasons = Explainer.format(results, n)
        self.add_history(self._instance, self.__class__.__name__, self.minimal_contrastive_reason.__name__, reasons)
        return reasons


    def sufficient_reason(self, *, time_limit=None):
        """A sufficient reason (also known as prime implicant explanation) for an instance x given a class described by a Boolean function f is a
        subset t of the characteristics of x that is minimal w.r.t. set inclusion such that any instance x' sharing this set t of characteristics
        is classified by f as x is.

        Returns:
            (obj:`list` of :obj:`int`): Reason in the form of literals (binary form). The to_features() method allows to obtain the features of this
             reason.
        """
        if self._instance is None:
            raise ValueError("Instance is not set")

        if self._random_forest.n_classes == 2:
            hard_clauses = self._random_forest.to_CNF(self._instance, self._binary_representation, self.target_prediction, tree_encoding=Encoding.MUS)
        else:
            hard_clauses = self._random_forest.to_CNF_sufficient_reason_multi_classes(self._instance, self.binary_representation,
                                                                                      self.target_prediction)

        if self._theory:
            hard_clauses = hard_clauses + tuple(self._random_forest.get_theory(self._binary_representation))

        # Check if excluded features produce a SAT problem => No sufficient reason
        if len(self._excluded_literals) > 0:
            SATSolver = GlucoseSolver()
            SATSolver.add_clauses(hard_clauses)
            SATSolver.add_clauses([[lit] for lit in self._binary_representation if self._is_specific(lit)])
            result, time = SATSolver.solve(time_limit=time_limit)
            if result is not None:
                return None

        hard_clauses = list(hard_clauses)
        soft_clauses = tuple((lit,) for lit in self._binary_representation if self._is_specific(lit))
        mapping = [0 for _ in range(len(self._binary_representation) + 2)]
        i = 2  # first good group for literals is 2 (1 is for the CNF representing the RF.
        for lit in self._binary_representation:
            if self._is_specific(lit):
                mapping[i] = lit
                i += 1
        n_variables = CNFencoding.compute_n_variables(hard_clauses)
        muser_solver = MUSERSolver()
        muser_solver.write_gcnf(n_variables, hard_clauses, soft_clauses)
        model, status, self._elapsed_time = muser_solver.solve(time_limit)
        reason = [mapping[i] for i in model if i > 1]

        reason = Explainer.format(reason, 1)
        self.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__, reason)
        return reason


    def minimal_sufficient_reason(self, time_limit=None):
        """A sufficient reason (also known as prime implicant explanation) for an instance x given a class described by a Boolean function f is a
         subset t of the characteristics of x that is minimal w.r.t. set inclusion such that any instance x' sharing this set t of characteristics
         is classified by f as x is.

        This method compute minimal sufficient reason. A minimal sufficient reason for x given f is a sufficient reason for x given f that contains
        a minimal number of literals.

        Returns:
            (obj:`list` of :obj:`int`): Reason in the form of literals (binary form). The to_features() method allows to obtain the features of
            this reason.
        """

        if self._instance is None:
            raise ValueError("Instance is not set")

        if self._random_forest.n_classes == 2:
            hard_clauses = self._random_forest.to_CNF(self._instance, self._binary_representation, self.target_prediction, tree_encoding=Encoding.MUS)
        else:
            hard_clauses = self._random_forest.to_CNF_sufficient_reason_multi_classes(self._instance, self.binary_representation,
                                                                                      self.target_prediction)

        if self._theory:
            clauses_theory = self._random_forest.get_theory(self._binary_representation)
            hard_clauses = hard_clauses + tuple(clauses_theory)

        if len(self._excluded_literals) > 0:
            SATSolver = GlucoseSolver()
            SATSolver.add_clauses(hard_clauses)
            SATSolver.add_clauses([[lit] for lit in self._binary_representation if self._is_specific(lit)])
            result = SATSolver.solve(time_limit=time_limit)
            if result is not None:
                return None

        soft_clauses = tuple((lit,) for lit in self._binary_representation if self._is_specific(lit))
        optux_solver = OPTUXSolver()
        optux_solver.add_hard_clauses(hard_clauses)
        optux_solver.add_soft_clauses(soft_clauses, weight=1)
        reason = optux_solver.solve(self._binary_representation)

        reason = Explainer.format(reason)
        self.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__, reason)
        return reason


    def majoritary_reason(self, *, n=1, n_iterations=50, time_limit=None, seed=0):
        """Informally, a majoritary reason for classifying a instance x as positive by some random forest f
        is a prime implicant t of a majority of decision trees in f that covers x. (see the Trading Complexity for Sparsity
        in Random Forest Explanations paper (Gilles Audemard, Steve Bellart, Louenas Bounia, Frederic Koriche,
        Jean-Marie Lagniez and Pierre Marquis1) for more information)

        Args:
            n (int|ALL, optional): The desired number of reasons. Defaults to 1, currently needs to be 1 or Exmplainer.ALL.
            time_limit (int, optional): The maximum time to compute the reasons. None to have a infinite time. Defaults to None.
        """

        if self._instance is None:
            raise ValueError("Instance is not set")

        reason_expressivity = ReasonExpressivity.Conditions
        if seed is None: seed = -1
        if isinstance(n, int) and n == 1:
            if self.c_RF is None:
                # Preprocessing to give all trees in the c++ library
                self.c_RF = c_explainer.new_classifier_RF(self._random_forest.n_classes)
                for tree in self._random_forest.forest:
                    try:
                        c_explainer.add_tree(self.c_RF, tree.raw_data_for_CPP())
                    except Exception as e:
                        print("Erreur", str(e))
                        exit(1)

            if time_limit is None:
                time_limit = 0
            implicant_id_features = ()  # FEATURES : TODO
            c_explainer.set_excluded(self.c_RF, tuple(self._excluded_literals))
            if self._theory:
                c_explainer.set_theory(self.c_RF, tuple(self._random_forest.get_theory(self._binary_representation)))
            current_time = time.process_time()
            reason = c_explainer.compute_reason(self.c_RF, self._binary_representation, implicant_id_features, self.target_prediction, n_iterations,
                                                time_limit, int(reason_expressivity), seed)
            total_time = time.process_time() - current_time
            self._elapsed_time = total_time if time_limit == 0 or total_time < time_limit else Explainer.TIMEOUT
            if reason_expressivity == ReasonExpressivity.Features:
                reason = self.to_features_indexes(reason)  # TODO

            reason = Explainer.format(reason)
            self.add_history(self._instance, self.__class__.__name__, self.majoritary_reason.__name__, reason)
            return reason

        if self._theory:
            raise NotImplementedError("Theory and all majoritary is not yet implanted")
        n = n if type(n) == int else float('inf')

        clauses = self._random_forest.to_CNF(self._instance, self._binary_representation, self.target_prediction, tree_encoding=Encoding.SIMPLE)
        max_id_variable = CNFencoding.compute_max_id_variable(self._binary_representation)
        solver = GlucoseSolver()

        solver.add_clauses([[lit for lit in clause if lit in self._binary_representation or abs(lit) > max_id_variable] for clause in clauses])

        if len(self._excluded_literals) > 0:
            solver.add_clauses([-lit] for lit in self._excluded_literals)

        majoritaries = []
        time_used = 0
        while True:
            result, _time = solver.solve()
            time_used += _time
            if result is None or (time_limit is not None and time_used > time_limit) or (time_limit is None and len(majoritaries) >= n):
                break

            majoritary = [lit for lit in result if lit in self._binary_representation]
            if majoritary not in majoritaries:
                majoritaries.append(majoritary)
            solver.add_clauses([[-lit for lit in result if abs(lit) <= max_id_variable]])  # block this implicant
        self._elapsed_time = time_used if (time_limit is None or time_used < time_limit) else Explainer.TIMEOUT
        reasons = Explainer.format(CNFencoding.remove_subsumed(majoritaries), n)
        self.add_history(self._instance, self.__class__.__name__, self.majoritary_reason.__name__, reasons)
        return reasons


    def preferred_majoritary_reason(self, *, method, n=1, time_limit=None, weights=None, features_partition=None):
        """This approach consists in exploiting a model, making precise her / his
        preferences about reasons, to derive only preferred reasons. See the On Preferred Abductive Explanations for Decision Trees and Random Forests
         paper (Gilles Audemard, Steve Bellart, Louenas Bounia, Frederic Koriche, Jean-Marie Lagniez and Pierre Marquis) for more details.

        Several methods are available thought the method parameter:
          `PreferredReasonMethod.MINIMAL`: Equivalent to minimal_majoritary_reason().
          `PreferredReasonMethod.WEIGHTS`: Weight given bu the user thought the weights parameters.
          `PreferredReasonMethod.SHAPELY`: Shapely values for the model trained on data. Only available with a model from scikitlearn or a save from
          scikitlearn (not compatible with a generic save). For each random forest F, the opposite of the SHAP score
           [Lundberg and Lee, 2017; Lundberg et al., 2020] of each feature of the instance x at hand given F computed using SHAP
           (shap.readthedocs.io/en/latest/api.html)
          `PreferredReasonMethod.FEATURE_IMPORTANCE`: The opposite of the f-importance of each feature in F as computed by Scikit-Learn
          [Pedregosa et al., 2011]. Only available with a model from scikitlearn or a save from scikitlearn (not compatible with a generic save).
          `PreferredReasonMethod.WORD_FREQUENCY`: The opposite of the Zipf frequency of each feature viewed as a word in the wordfreq library.
          `PreferredReasonMethod.WORD_FREQUENCY_LAYERS`: WORD_FREQUENCY with layers.

        Args:
            n (int|ALL, optional): The desired number of reasons. Defaults to Explainer.ALL.
            time_limit (int, optional): The maximum time to compute the reasons. None to have a infinite time. Defaults to None.
            method (Explainer.MINIMAL|Explainer.WEIGHTS|, optional): _description_. Defaults to None.
            weights (:obj:`list` of `int`|:obj:`list`, optional): Can be a list representing the weight of each feature or a dict where a key is
            a feature and the value its weight. Defaults to None.

        Returns:
            _type_: _description_
        """

        if self._instance is None:
            raise ValueError("Instance is not set")

        n = n if type(n) == int else float('inf')

        if self._random_forest.n_classes == 2:
            clauses = self._random_forest.to_CNF(self._instance, self._binary_representation, self.target_prediction, tree_encoding=Encoding.SIMPLE)
        else:
            clauses = self._random_forest.to_CNF_majoritary_reason_multi_classes(self._instance, self._binary_representation, self.target_prediction)

        n_variables = CNFencoding.compute_n_variables(clauses)
        id_features = self._random_forest.get_id_features(self._binary_representation)
        
        weights = compute_weight(method, self._instance, weights, self._random_forest.forest[0].learner_information,
                                 features_partition=features_partition)
        solver = OPENWBOSolver()
        max_id_variable = CNFencoding.compute_max_id_variable(self._binary_representation)
        map_abs_implicant = [0 for _ in range(0, n_variables + 1)]
        for lit in self._binary_representation:
            map_abs_implicant[abs(lit)] = lit
        # Hard clauses
        for c in clauses:
            solver.add_hard_clause([lit for lit in c if abs(lit) > max_id_variable or map_abs_implicant[abs(lit)] == lit])

        if self._theory:
            clauses_theory = self._random_forest.get_theory(self._binary_representation)
            for c in clauses_theory:
                solver.add_hard_clause(c)

        # excluded features
        for lit in self._binary_representation:
            if not self._is_specific(lit):
                solver.add_hard_clause([-lit])

        # Soft clauses
        for i in range(len(self._binary_representation)):
            solver.add_soft_clause([-self._binary_representation[i]], weights[id_features[i] - 1])

        # Solving
        time_used = 0
        best_score = -1
        reasons = []
        first_call = True

        while True:
            status, model, _time = solver.solve(time_limit=None if time_limit is None else time_limit - time_used)
            time_used += _time
            if model is None:
                if first_call:
                    return ()
                reasons = Explainer.format(reasons, n)
                if method == PreferredReasonMethod.Minimal:
                    self.add_history(self._instance, self.__class__.__name__, self.minimal_majoritary_reason.__name__, reasons)
                else:
                    self.add_history(self._instance, self.__class__.__name__, self.preferred_majoritary_reason.__name__, reasons)
                return reasons

            prefered_reason = [lit for lit in model if lit in self._binary_representation]
            solver.add_hard_clause([-lit for lit in model if abs(lit) <= max_id_variable])

            # Compute the score
            score = numpy.sum([weights[id_features[abs(lit) - 1] - 1] for lit in prefered_reason])
            if first_call:
                best_score = score
            elif score != best_score:
                reasons = Explainer.format(reasons, n)
                if method == PreferredReasonMethod.Minimal:
                    self.add_history(self._instance, self.__class__.__name__, self.minimal_majoritary_reason.__name__, reasons)
                else:
                    self.add_history(self._instance, self.__class__.__name__, self.preferred_majoritary_reason.__name__, reasons)
                return reasons
            first_call = False

            reasons.append(prefered_reason)
            if (time_limit is not None and time_used > time_limit) or len(reasons) == n:
                break
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        reasons = Explainer.format(reasons, n)
        if method == PreferredReasonMethod.Minimal:
            self.add_history(self._instance, self.__class__.__name__, self.minimal_majoritary_reason.__name__, reasons)
        else:
            self.add_history(self._instance, self.__class__.__name__, self.preferred_majoritary_reason.__name__, reasons)
        return reasons


    def minimal_majoritary_reason(self, *, n=1, time_limit=None):
        return self.preferred_majoritary_reason(method=PreferredReasonMethod.Minimal, n=n, time_limit=time_limit)


    def is_majoritary_reason(self, reason, n_samples=50):
        extended_reason = self.extend_reason_with_theory(reason)
        if not self.is_implicant(extended_reason):
            return False
        tmp = list(reason)
        random.shuffle(tmp)
        nb = 0
        for lit in tmp:
            copy_reason = list(reason).copy()
            copy_reason.remove(lit)
            if self.is_implicant(tuple(copy_reason)):
                return False
            nb += 1
            if nb > n_samples:
                break
        return True
