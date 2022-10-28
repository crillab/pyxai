import time

from pyxai.sources.core.explainer.Explainer import Explainer
from pyxai.sources.core.structure.decisionTree import DecisionTree
from pyxai.sources.core.structure.type import PreferredReasonMethod
from pyxai.sources.core.tools.encoding import CNFencoding
from pyxai.sources.core.tools.utils import compute_weight
from pyxai.sources.solvers.COMPILER.D4Solver import D4Solver
from pyxai.sources.solvers.MAXSAT.OPENWBOSolver import OPENWBOSolver
from pyxai.sources.solvers.SAT.glucoseSolver import GlucoseSolver


class ExplainerDT(Explainer):

    def __init__(self, tree, instance=None):
        """Create object dedicated to finding explanations from a decision tree ``tree`` and an instance ``instance``.

        Args:
            tree (DecisionTree): The model in the form of a DecisionTree object.
            instance (:obj:`list` of :obj:`int`, optional): The instance (an observation) on which explanations must be calculated. Defaults to None.
        """
        super().__init__()
        self._tree = tree  # The decision _tree.
        if instance is not None:
            self.set_instance(instance)


    @property
    def tree(self):
        return self._tree


    def _to_binary_representation(self, instance):
        return self._tree.instance_to_binaries(instance)


    def is_implicant(self, binary_representation):
        return self._tree.is_implicant(binary_representation, self.target_prediction)


    def predict(self, instance):
        return self._tree.predict_instance(instance)


    def to_features(self, binary_representation, *, eliminate_redundant_features=True, details=False):
        """_summary_

        Args:
            binary_representation (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self._tree.to_features(binary_representation, details=details, eliminate_redundant_features=eliminate_redundant_features)


    def direct_reason(self):
        """

        Returns:
            _type_: _description_
        """
        self._elapsed_time = 0
        direct_reason = self._tree.direct_reason(self._instance)
        if any(not self._is_specific(lit) for lit in direct_reason):
            return None  # The reason contains excluded features
        return Explainer.format(direct_reason)


    def contrastive_reason(self, *, n=1):
        self._elapsed_time = 0
        cnf = self._tree.to_CNF(self._instance)
        core = CNFencoding.extract_core(cnf, self._binary_representation)
        core = [c for c in core if all(self._is_specific(lit) for lit in c)]  # remove excluded
        contrastives = sorted(core, key=lambda clause: len(clause))
        return Explainer.format(contrastives, n) if type(n) != int else Explainer.format(contrastives[:n], n)


    def necessary_literals(self):
        self._elapsed_time = 0
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        core = CNFencoding.extract_core(cnf, self._binary_representation)
        # DO NOT remove excluded features. If they appear, they explain why there is no sufficient
        return sorted({lit for _, clause in enumerate(core) if len(clause) == 1 for lit in clause})


    def relevant_literals(self):
        self._elapsed_time = 0
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        core = CNFencoding.extract_core(cnf, self._binary_representation)
        return [lit for _, clause in enumerate(core) if len(clause) > 1 for lit in clause if self._is_specific(lit)]  # remove excluded features


    def _excluded_features_are_necesssary(self, prime_cnf):
        for lit in prime_cnf.necessary:
            if self._is_specific(lit) is False:
                return True
        return False


    def sufficient_reason(self, *, n=1, time_limit=None):
        time_used = 0
        n = n if type(n) == int else float('inf')
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        prime_implicant_cnf = CNFencoding.to_prime_implicant_CNF(cnf, self._binary_representation)

        if self._excluded_features_are_necesssary(prime_implicant_cnf):
            self._elapsed_time = 0
            return None

        SATsolver = GlucoseSolver()
        SATsolver.add_clauses(prime_implicant_cnf.cnf)

        # Remove excluded features
        SATsolver.add_clauses([[-prime_implicant_cnf.from_original_to_new(lit)] for lit in self._excluded_literals])

        sufficient_reasons = []
        while True:
            if (time_limit is not None and time_used > time_limit) or len(sufficient_reasons) == n:
                break
            result, _time = SATsolver.solve(None if time_limit is None else time_limit - time_used)
            time_used += _time
            if result is None:
                break
            sufficient_reasons.append(prime_implicant_cnf.get_reason_from_model(result))
            SATsolver.add_clauses([prime_implicant_cnf.get_blocking_clause(result)])
        self._elapsed_time = time_used if (time_limit is None or time_used < time_limit) else Explainer.TIMEOUT
        return Explainer.format(sufficient_reasons, n)


    def preferred_sufficient_reason(self, *, method, n=1, time_limit=None, weights=None, features_partition=None):
        n = n if type(n) == int else float('inf')
        cnf = self._tree.to_CNF(self._instance)
        self._elapsed_time = 0

        prime_implicant_cnf = CNFencoding.to_prime_implicant_CNF(cnf, self._binary_representation)

        # excluded are necessary => no reason
        if self._excluded_features_are_necesssary(prime_implicant_cnf):
            return None

        cnf = prime_implicant_cnf.cnf
        if len(cnf) == 0:  # TODO: test when this case append
            return [lit for lit in prime_implicant_cnf.necessary]

        weights = compute_weight(method, self._instance, weights, self._tree.learner_information, features_partition=features_partition)
        weights_per_feature = {i + 1: weight for i, weight in enumerate(weights)}

        soft = [lit for lit in prime_implicant_cnf.mapping_original_to_new if lit != 0]
        weights_soft = []
        for lit in soft:  # soft clause
            for i in range(len(self._instance)):
                if self._tree.to_features([lit], eliminate_redundant_features=False, details=True)[0]["id"] == i + 1:
                    weights_soft.append(weights[i])

        solver = OPENWBOSolver()

        # Hard clauses
        for c in cnf:
            solver.add_hard_clause(c)

        # Soft clauses
        for i in range(len(soft)):
            solver.add_soft_clause([-soft[i]], weights_soft[i])

        # Remove excluded features
        for lit in self._excluded_literals:
            solver.add_hard_clause([-prime_implicant_cnf.from_original_to_new(lit)])

        # Solving
        time_used = 0
        best_score = -1
        reasons = []
        first_call = True

        while True:
            status, model, _time = solver.solve(time_limit=0 if time_limit is None else time_limit - time_used)
            time_used += _time
            if model is None:
                break

            preferred = prime_implicant_cnf.get_reason_from_model(model)
            solver.add_hard_clause(prime_implicant_cnf.get_blocking_clause(model))
            # Compute the score
            score = sum([weights_per_feature[feature["id"]] for feature in
                         self._tree.to_features(preferred, eliminate_redundant_features=False, details=True)])
            if first_call:
                best_score = score
            elif score != best_score:
                break
            first_call = False
            reasons.append(preferred)
            if (time_limit is not None and time_used > time_limit) or len(reasons) == n:
                break
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        return Explainer.format(reasons, n)


    def minimal_sufficient_reason(self, *, n=1, time_limit=None):
        return self.preferred_sufficient_reason(method=PreferredReasonMethod.Minimal, n=n, time_limit=time_limit)


    def n_sufficient_reasons_per_attribute(self, *, time_limit=None):
        cnf = self._tree.to_CNF(self._instance)
        prime_implicant_cnf = CNFencoding.to_prime_implicant_CNF(cnf, self._binary_representation)

        if self._excluded_features_are_necesssary(prime_implicant_cnf):
            self._elapsed_time = 0
            return None

        if len(prime_implicant_cnf.cnf) == 0:  # Special case where all in necessary
            return {lit: 1 for lit in prime_implicant_cnf.necessary}

        compiler = D4Solver()
        # Remove excluded features
        cnf = list(prime_implicant_cnf.cnf)
        for lit in self._excluded_literals:
            cnf.append([-prime_implicant_cnf.from_original_to_new(lit)])

        compiler.add_cnf(cnf, prime_implicant_cnf.n_literals - 1)
        compiler.add_count_model_query(cnf, prime_implicant_cnf.n_literals - 1, prime_implicant_cnf.n_literals_mapping)

        time_used = -time.time()
        n_models = compiler.solve(time_limit)
        time_used += time.time()

        self._elapsed_time = Explainer.TIMEOUT if n_models[1] == -1 else time_used

        n_necessary = n_models[0] if len(n_models) > 0 else 1

        n_sufficients_per_attribute = {n: n_necessary for n in prime_implicant_cnf.necessary}
        for lit in range(1, prime_implicant_cnf.n_literals_mapping):
            n_sufficients_per_attribute[prime_implicant_cnf.mapping_new_to_original[lit]] = n_models[lit]
        return n_sufficients_per_attribute


    def is_reason(self, reason, *, n_samples=-1):
        return self._tree.is_implicant(reason, self.target_prediction)


    @staticmethod
    def rectify(_tree, positive_rectifying__tree, negative_rectifying__tree):
        not_positive_rectifying__tree = positive_rectifying__tree.negating__tree()
        not_negative_rectifying__tree = negative_rectifying__tree.negating__tree()

        _tree_1 = positive_rectifying__tree.concatenate__tree(not_negative_rectifying__tree)
        _tree_2 = negative_rectifying__tree.concatenate__tree(not_positive_rectifying__tree)

        not__tree_2 = _tree_2.negating__tree()

        _tree_and_not__tree_2 = _tree.concatenate__tree(not__tree_2)
        _tree_and_not__tree_2.simplify()

        _tree_and_not__tree_2_or__tree_1 = _tree_and_not__tree_2.disjoint__tree(_tree_1)
        _tree_and_not__tree_2_or__tree_1.simplify()
        return _tree_and_not__tree_2_or__tree_1
