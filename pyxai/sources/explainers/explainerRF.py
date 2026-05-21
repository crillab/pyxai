import random
import time

import c_explainer
import numpy

from pyxai.sources.explainers.Explainer import Explainer
from pyxai.sources.type import Encoding, PreferredReasonMethod, TypeTheory, ReasonExpressivity
from pyxai.sources.tools.encoding import CNFencoding
from pyxai.sources.tools.utils import compute_weight
from pyxai.sources.solvers.MAXSAT.OPENWBOSolver import OPENWBOSolver
from pyxai.sources.solvers.MUS.MUSERSolver import MUSERSolver
from pyxai.sources.solvers.MUS.OPTUXSolver import OPTUXSolver
from pyxai.sources.solvers.SAT.glucoseSolver import GlucoseSolver
from pyxai import Tools

class ExplainerRF(Explainer):
    r"""The Random Forest explainer.

    This class adapts the generic ```Explainer```  to a ```RandomForest``` model. It provides methods to:

    - set an instance and derive its Boolean (literal-based) representation,
    - compute different families of explanations (direct, contrastive, sufficient, majoritary),
    - optionally *rectify* (modify) the random forest to enforce a desired label under given conditions.

    Some functions come from the parent class ```Explainer```, their documentation is available in the dedicated page:
    - Main Methods: ```set_instance```, ```set_features_type```, ```get_model```, ```predict```, ```activate_theory```, ```deactivate_theory```, ```set_excluded_features```, ```unset_excluded_features```;
    - Human-Readable Explanation Methods: ```to_features```, ```get_feature_names```, ```get_feature_names_from_literal```, ```reason_contains_features```;
    - Explanation verification methods: ```is_implicant```, ```is_reason```, ```is_sufficient_reason```, ```is_contrastive_reason```.

    See Also
    --------
    - The [documentation pages](https://www.cril.univ-artois.fr/pyxai/documentation/explanations/RFexplanations/)
    - The paper [Trading Complexity for Sparsity in Random Forest Explanations paper](https://arxiv.org/abs/2108.05276)

    Notes
    -----
    Some methods are limited to binary classification (2 classes). When multi-class support
    is not available, an exception `NotImplementedError` is raised.
    """


    def __init__(self, random_forest, instance=None):
        """
        Initialize an explainer for a Random Forest model.

        Parameters
        ----------
        random_forest: ```RandomForest```
            The RandomForest model to explain.
        instance : ```list[int]``` (optional, default: None)
            The instance (observation) for which explanations will be computed.
            If ``None``, you must call `set_instance` before requesting explanations.


        """
        super().__init__()

        self._random_forest = random_forest
        self.c_RF = None
        self.c_rectifier = None

        if self._random_forest.class_names is not None:
            self._class_names = numpy.array(self._random_forest.class_names)

        if instance is not None:
            self.set_instance(instance)


    @property
    def random_forest(self):
        return self._random_forest


    def to_features(self,
                    binary_representation,
                    *, eliminate_redundant_features=True,
                    details=False,
                    contrastive=False,
                    without_intervals=False):
        r"""Converts a binary representation of a reason into the features space.

                When the parameter details is set to False, returns a Tuple of String where each String represents a condition
                "\<id_feature\> \<operator\> \<threshold\>" associated with the binary representation given as first parameter. By default, a string represents
                such a condition but if you need more information, you can set the parameter details to True. In this case, the method returns
                a Tuple of Dict where each dictionary provides more information on the condition. This method also allows one to eliminate redundant
                conditions. For example, if we have "feature_a > 3" and "feature_a > 2", we keep only the binary variable linked to the Boolean corresponding
                to the "feature_a > 3". Therefore, if you want to get all conditions, you have to set the parameter eliminate_redundant to False.

                Must be implemented by subclasses.

                Parameters
                ----------
                binary_representation: list[int]
                    the binary representation to convert

                eliminate_redundant_features: ```bool``` (optional, default: True)
                    eliminate possible redundant features.

                details: ```bool``` (optional, default: False)
                    returns a detailed representation instead of a string

                contrastive:  ```bool``` (optional, default: False):
                    invert logic in order to explain a contrastive reason

                Returns
                -------
                ```list[str|dict]:
                    a list of string or of dictionaries depending the value of details.

                See Also
                --------
                - [Documentation page](http://localhost:4000/pyxai/documentation/explainer/concepts/)
        """
        return self._random_forest.to_features(binary_representation, eliminate_redundant_features=eliminate_redundant_features, details=details,
                                               contrastive=contrastive, without_intervals=without_intervals, feature_names=self.get_feature_names())


    def _to_binary_representation(self, instance):
        return self._random_forest.instance_to_binaries(instance)

    def is_implicant(self, binary_representation, *, prediction=None):
        """
        Check whether a binary representation (partial or complete) implies the prediction.

        Parameters
        ----------
        binary_representation: (```list[int]```)
            the binary representation

        Returns
        -------
        ```bool```:
            True if the binary representation is an implicant of the selected instance, False otherwise.
        """
        if prediction is None:
            prediction = self.cpp_target_prediction
        
        binary_representation = self.extend_reason_with_theory(binary_representation)
        return self._random_forest._is_implicant(binary_representation, prediction)

    def predict_votes(self, instance):
        return self._random_forest.predict_votes(instance)

    def predict(self, instance):
        r"""Predict the label/value for an instance using the underlying tree.

        Parameters
        ----------
        instance: ```list[float]```
            The instance to classify

        Returns
        -------
        ```int```:
            The  prediction for the given instance.
        """
        return self._random_forest.predict_instance(instance)


    def direct_reason(self):
        r"""Compute the direct reason for the current instance.

        The direct reason corresponds to a subset of the binary representation obtained by collecting the root-to-leaf
        path conditions (in binary/literal form) that cover the instance across the forest.
        If the result contains excluded features, None is returned.

        Returns
        -------
        ```list[int]```:
            A list of literals representing the direct reason.
        ```None```:
            If excluded features appear in the direct reason.

        Raises
        ------
        ValueError
            If no instance has been set.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/RFexplanations/direct/)
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
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.direct_reason.__name__, reason)
        return reason


    def minimal_contrastive_reason(self, *, n=1, time_limit=None):
        """Compute up to ``n`` contrastive reasons for the current instance.

        Contrastive explanations explain why the instance has not been classified by the ML model as expected.

        Parameters
        ----------
        n: ```int``` | ```Explaining.ALL```  (optional,  default=```1```)
            Maximum number of contrastive reasons to return. If a ```Explaining.ALL``` is provided, all computed candidates are formatted/returned.
        time_limit: ```float``` (optional,  default=```None```)
            Time limit (seconds) for the whole enumeration. If reached, `elapsed_time` is set to ```Explaining.TIMEOUT```.

        Returns
        -------
        ```tuple[int]```:
            if n == 1, returns a single reason as a tuple of literals.
        ```tuple[tuple[int]]```:
            If n > 1, returns a tuple of reasons (each one is a tuple of literals).
        ```None```:
            If all reasons contain excluded features.

        Raises
        ------
        ValueError
            If no instance has been set.

        See Also
        --------
        - [Documentation page](http://www.cril.univ-artois.fr/pyxai/documentation/classification/RFexplanations/contrastive/)
        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        if self._random_forest.n_classes > 2:
            raise NotImplementedError("Minimal contrastive reason is not implemented for the multi class case")
        n = n if type(n) == int else float('inf')
        first_call = True
        time_limit = 0 if time_limit is None else time_limit
        best_score = 0
        tree_cnf = self._random_forest.to_CNF(self._instance, self._binary_representation, target_prediction=1 if self.cpp_target_prediction == 0 else 0,
                                              tree_encoding=Encoding.SIMPLE)

        # structure to help to do this method faster
        map_in_binary_representation = dict()
        for id in self._binary_representation:
            map_in_binary_representation[id] = True
            map_in_binary_representation[-id] = False

        max_id_binary_representation = CNFencoding.compute_max_id_variable(self._binary_representation)

        max_id_binary_cnf = CNFencoding.compute_max_id_variable(tree_cnf)

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
                theory_type=TypeTheory.NewVariables,
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
                Tools.verbose("minimal_contrastive_reason: end by time_limit or 'n' reached.") 
                break

        self._elapsed_time = time_used if time_limit == 0 or time_used < time_limit else Explainer.TIMEOUT

        reasons = Explainer.format(results, n)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.minimal_contrastive_reason.__name__, reasons)
        return reasons


    def sufficient_reason(self, *, time_limit=None):
        r"""Compute a sufficient reason.

        A *sufficient reason* is a (typically) minimal subset of the binary representation that still entails the
        current prediction.

        Parameters
        ----------
        time_limit: ```float | None```, optional
            Time limit (seconds) for the whole enumeration. If reached, property elapsed_time is set to ```Explaining.TIMEOUT```.

        Returns
        -------
        ```list[int]```:
            The sufficient reason.
        
        ```None```:
            If excluded features are necessary.

        Raises
        ------
        ValueError
            If no instance has been set.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/RFexplanations/sufficient/)
        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        
        if self._random_forest.n_classes == 2:
            hard_clauses = self._random_forest.to_CNF(self._instance, self._binary_representation, self.cpp_target_prediction, tree_encoding=Encoding.MUS)
        else:
            hard_clauses = self._random_forest.to_CNF_sufficient_reason_multi_classes(self._instance, self.binary_representation, self.cpp_target_prediction)

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
        if model is None: 
            return None
        reason = [mapping[i] for i in model if i > 1]

        reason = Explainer.format(reason, 1)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__, reason)
        return reason


    def minimal_sufficient_reason(self, time_limit=None):
        r"""Compute a *minimal-size* sufficient reason for the current instance.

        This is an optimization variant`sufficient_reason` that aims to minimize the number of literals in the returned sufficient reason.

        Parameters
        ----------
        time_limit: ```float | None```, optional
            Time limit (seconds) for the whole enumeration. If reached, property elapsed_time is set
            to ```Explaining.TIMEOUT```.

        Returns
        -------
        ```list[int]```:
            A minimal sufficient reason.
        ```None```
            if time limit is reached or if excluded features are necessary).

        Raises
        ------
        ValueError
            If no instance has been set.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/RFexplanations/sufficient/)

        Warning
        -------
        Computing a minimal sufficient reason is computationally very hard. Thus, it is possible only for small dataset.

        """
        if self._instance is None:
            raise ValueError("Instance is not set")

        if self._random_forest.n_classes == 2:
            hard_clauses = self._random_forest.to_CNF(self._instance, self._binary_representation, self.cpp_target_prediction, tree_encoding=Encoding.MUS)
        else:
            hard_clauses = self._random_forest.to_CNF_sufficient_reason_multi_classes(self._instance, self.binary_representation, self.cpp_target_prediction)

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
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__, reason)
        return reason


    def majoritary_reason(self, *, n=1, n_iterations=50, time_limit=None, seed=0):
        """Compute a majoritary reason for the current instance.

        Informally, a majoritary reason is a subset of the binary representation  that covers a majoritary
        of trees.
        Returns n majoritary reasons sorted by increasing of the size for the current instance in a Tuple
        (when n is set to 1, does not return a Tuple but just the reason). Supports excluded features.

        Parameters
        ----------
        n: ```int``` | ```Explaining.ALL```, (optional, default=```1```)
            Maximum number of majoritary reasons to enumerate. If ```Explaining.ALL```, treated as unbounded.
        n_iterations: ```int``` (optional, default: ```50```)
            Number of randomized iterations used to extract the smallest one.

        time_limit: ```float```, (optional, default: ```None```)
            Time limit (seconds) for the whole enumeration. If reached, property ```elapsed_time``` is set
            to ```Explaining.TIMEOUT```.

        seed: ```int```, (optional, default=```0```)
            The seed when the greedy algorithm is used. Set to 0 this parameter to use a random seed. The default value is 0.
        n_iterations: ```int```, (optional, default: ```50```)
            Only used if n=1. It is the number of iterations done by the greedy algorithm.

        Returns
        -------
        ```tuple[int]```:
            if n == 1, returns a single reason as a tuple of literals.
        ```tuple[tuple[int]]```:
            If n > 1, returns a tuple of reasons (each one is a tuple of literals).
        ```None```:
            If all reasons contain excluded features.

        Note
        ----
        - When n=1, a greedy algorithm (in polynomial time) is called to extract a majoritary reason. n_iterations allows to call the
        olgorithm several time  to extract the smallest of those calculated.
        - When n > 1, a different algorithm is called.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/RFexplanations/majoritary/)

        Raises
        ------
        ValueError
            If no instance has been set.
        NotImplementedError
            If theory is enabled while requesting enumeration (implementation limitation).
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
            reason = c_explainer.compute_reason(self.c_RF, self._binary_representation, implicant_id_features, self.cpp_target_prediction, n_iterations,
                                                time_limit, reason_expressivity, seed, 0)
            total_time = time.process_time() - current_time
            self._elapsed_time = total_time if time_limit == 0 or total_time < time_limit else Explainer.TIMEOUT
            if reason_expressivity == ReasonExpressivity.Features:
            
                reason = self.to_features_indexes(reason)  # TODO

            reason = Explainer.format(reason)
            self._visualisation.add_history(self._instance, self.__class__.__name__, self.majoritary_reason.__name__, reason)
            return reason

        if self._theory:
            raise NotImplementedError("Theory and all majoritary is not yet implanted")
        n = n if type(n) == int else float('inf')

        clauses = self._random_forest.to_CNF(self._instance, self._binary_representation, self.cpp_target_prediction, tree_encoding=Encoding.SIMPLE)
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
    
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.majoritary_reason.__name__, reasons)
        return reasons


    def preferred_majoritary_reason(self, *, method, n=1, time_limit=None, weights=None, features_partition=None):
        r"""Compute sufficient reasons optimized by a preference criterion.

        This approach consists in exploiting a model, making precise her / his
        preferences about reasons, to derive only preferred reasons. See the On Preferred Abductive Explanations for Decision Trees and Random Forests
        paper (Gilles Audemard, Steve Bellart, Louenas Bounia, Frederic Koriche, Jean-Marie Lagniez and Pierre Marquis) for more details.

        Parameters
        ----------
        method : ```PreferredReasonMethod``` | ```str```
            The preference method used to discriminate reasons. 
            Possible values are defined in the ```PreferredReasonMethod``` enum.

        n: ```int``` | ```Explaining.ALL``` (optional, default=```1```)
            Maximum number of majoritary reasons to enumerate. If ```Explaining.ALL```, treated as unbounded.

        time_limit: ```float``` (optional, default=```None```)
            Time limit (seconds) for the whole enumeration. If reached, property ```elapsed_time``` is set
            to ```Explaining.TIMEOUT```.

        weights : list[float] (optional, default=```None```)
            Per-feature weights.
            If ```None```, weights are computed from using the parameter method.
        
        features_partition: ```object``` (optional, default=```None```)
            Partitioning information passed to the weight computation helper.

        Returns
        -------
        ```tuple[int]```:
            if n == 1, returns a single reason as a tuple of literals.
        ```tuple[tuple[int]]```:
            If n > 1, returns a tuple of reasons (each one is a tuple of literals).
        ```None```:
            If all reasons contain excluded features.

        """

        if self._instance is None:
            raise ValueError("Instance is not set")
            
        n = n if type(n) == int else float('inf')

        if self._random_forest.n_classes == 2:
            clauses = self._random_forest.to_CNF(self._instance, self._binary_representation, self.cpp_target_prediction, tree_encoding=Encoding.SIMPLE)
        else:
            clauses = self._random_forest.to_CNF_majoritary_reason_multi_classes(self._instance, self._binary_representation, self.cpp_target_prediction)

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
                    self._visualisation.add_history(self._instance, self.__class__.__name__, self.minimal_majoritary_reason.__name__, reasons)
                else:
                    self._visualisation.add_history(self._instance, self.__class__.__name__, self.preferred_majoritary_reason.__name__, reasons)
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
                    self._visualisation.add_history(self._instance, self.__class__.__name__, self.minimal_majoritary_reason.__name__, reasons)
                else:
                    self._visualisation.add_history(self._instance, self.__class__.__name__, self.preferred_majoritary_reason.__name__, reasons)
                return reasons
            first_call = False

            reasons.append(prefered_reason)
            if (time_limit is not None and time_used > time_limit) or len(reasons) == n:
                break
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        reasons = Explainer.format(reasons, n)
    
        if method == PreferredReasonMethod.Minimal:
            self._visualisation.add_history(self._instance, self.__class__.__name__, self.minimal_majoritary_reason.__name__, reasons)
        else:
            self._visualisation.add_history(self._instance, self.__class__.__name__, self.preferred_majoritary_reason.__name__, reasons)
        return reasons


    def minimal_majoritary_reason(self, *, n=1, time_limit=None):
        r"""Compute up to ```n``` minimal majoritary reasons of minimal size.


        Returns up n minimal  majoritary reason of the current instance in a Tuple (when n is set to 1, does not return a Tuple but just the reason). Supports the excluded features.

        Parameters
        ----------
        n: ```int``` | ```Explaining.ALL```, (optional, default=```1```)
            Maximum number of majoritary reasons to enumerate. If ```Explaining.ALL```, treated as unbounded.
        time_limit: ```float``` (optional, default=```None```)
            Time limit (seconds) for the whole enumeration. If reached, elapsed_time is set
            to ```Explaining.TIMEOUT```.

        Returns
        -------
        ```tuple[int]```:
            if n == 1, returns a single reason as a tuple of literals.
        ```tuple[tuple[int]]```:
            If n > 1, returns a tuple of reasons (each one is a tuple of literals).
        ```None```:
            If all reasons contain excluded features.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/RFexplanations/majoritary/)
        """
        return self.preferred_majoritary_reason(method=PreferredReasonMethod.Minimal, n=n, time_limit=time_limit)


    def is_majoritary_reason(self, reason, n_samples=50):
        r"""Heuristically check whether a given reason behaves like a majoritary reason.

        The check proceeds in two steps:
        1) Verify that ``reason`` is an implicant (optionally extended with theory).
        2) Randomly try removing literals and test if the reduced set remains an implicant
                   (to detect non-minimality). Only up to ``n_samples`` removal attempts are made.

        Parameters
        ----------
        reason: ```list[int]```
            Candidate reason (literals).
        n_samples : ```int``` (optional, default:```50```)
            Maximum number of literal-removal trials.

        Returns
        -------
        ```bool```
            True if the reason passes the implicant and (heuristic) minimality checks, False otherwise.
        """
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
    
        

    def most_anchored_reason(self, *, time_limit=None, check=False, type_references="normal"):
        if self._random_forest.n_classes == 2:
            cnf = self._random_forest.to_CNF(self._instance, self._binary_representation, self.cpp_target_prediction, tree_encoding=Encoding.MUS)
            if self._theory:
                size = len(cnf)
                clauses_theory = self._random_forest.get_theory(self._binary_representation)
                cnf = cnf + tuple(clauses_theory)
                print("Theory enabled: clauses: "+ str(size) + " to " + str(len(cnf)))
                
                #for c in clauses_theory:
                #    cnf.append(c)
                    
            n_variables = CNFencoding.compute_max_id_variable(cnf)
            return self._most_anchored_reason(n_variables=n_variables, cnf=cnf, time_limit=time_limit, check=check, type_references=type_references)
        
        raise NotImplementedError("The anchored_reason() method for RF works only with binary-class datasets.")
        

    def rectify_cxx(self, *, conditions, label, tests=False):
        """
        C++ version
        Rectify the Decision Tree (self._tree) of the explainer according to a `conditions` and a `label`.
        Simplify the model (the theory can help to eliminate some nodes).

        Args:
            conditions (list or tuple): A decision rule in the form of list of literals (binary variables representing the conditions of the tree). 
            label (int): The label of the decision rule.   
        Returns:
            RandomForest: The rectified random forest.  
        """ 

        #check conditions and return a list of literals        
        conditions, change = self._random_forest.parse_conditions_for_rectify(conditions)
        if change is True:
            self.set_features_type(self._last_features_types)
         
        current_time = time.process_time()
        if self.c_rectifier is None:
            self.c_rectifier = c_explainer.new_rectifier()

        for i, tree in enumerate(self._random_forest.forest):
            c_explainer.rectifier_add_tree(self.c_rectifier, tree.raw_data_for_CPP())

        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - Initial (c++):", n_nodes_cxx) 

        # Rectification part
        c_explainer.rectifier_improved_rectification(self.c_rectifier, conditions, label)
        n_nodes_ccx =  c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After rectification (c++):", n_nodes_ccx)    
        if tests is True:
            for i in range(len(self._random_forest.forest)):
                tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, i)
                self._random_forest.forest[i].delete(self._random_forest.forest[i]._root)
                self._random_forest.forest[i]._root = self._random_forest.forest[i].from_tuples(tree_tuples)
            is_implicant = self.is_implicant(conditions, prediction=label)
            if is_implicant is False:
                raise ValueError("Problem: the rectified tree is not an implicant of the original tree!")
        

        # Simplify Theory part
        theory_cnf = self.get_model().get_theory(None)
        c_explainer.rectifier_set_theory(self.c_rectifier, tuple(theory_cnf))
        c_explainer.rectifier_simplify_theory(self.c_rectifier)

        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After simplification with the theory (c++):", n_nodes_cxx)

        if tests is True: 
            for i in range(len(self._random_forest.forest)):
                tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, i)
                self._random_forest.forest[i].delete(self._random_forest.forest[i]._root)
                self._random_forest.forest[i]._root = self._random_forest.forest[i].from_tuples(tree_tuples)
            is_implicant = self.is_implicant(conditions, prediction=label)
            if is_implicant is False:
                raise ValueError("Problem: the condition is not an imlicant of the prediction after simplify theory!")
        
        # Simplify part
        c_explainer.rectifier_simplify_redundant(self.c_rectifier)
        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After elimination of redundant nodes (c++):", n_nodes_cxx)
        
        # Get the C++ trees and convert it :) 
        for i in range(len(self._random_forest.forest)):
            tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, i)
            self._random_forest.forest[i].delete(self._random_forest.forest[i]._root)
            self._random_forest.forest[i]._root = self._random_forest.forest[i].from_tuples(tree_tuples)
        
        
        c_explainer.rectifier_free(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - Final (c++):", self._random_forest.n_nodes())
        if tests is True:
            is_implicant = self.is_implicant(conditions, prediction=label)
            if is_implicant is False:
                raise ValueError("Problem: the condition is not an imlicant of the prediction after simplify!")
        
        if self._instance is not None:
            self.set_instance(self._instance)

        self._elapsed_time = time.process_time() - current_time
        
        Tools.verbose("Rectification time:", self._elapsed_time)

        Tools.verbose("--------------")
        return self._random_forest

    def rectify(self, *, conditions, label, tests=False):
        r"""Rectify the random forest to satisfy a labeled constraint.

        Rectification enforces that the (parsed) ``conditions`` imply the given ``label`` in the model,
        then simplifies the resulting tree (optionally leveraging the explainer's theory constraints).

        Parameters
        ----------
        conditions: ```list[int]```
            Conditions (premises part of the classification rule) in the form of a binary representation (list of literals).
        label : ```int```
            Label representing the conclusion part of the classification rule.
        tests: ```bool``` (optional, default: False)
            If True, runs additional post-steps checks (can be expensive) to ensure rectification
            preserved the intended implicant property.

        Returns
        -------
        ```RandomForest```:
            The rectified (and simplified) decision tree.

        Raises
        ------
        ValueError: 
            If internal validation fails when ``tests=True``.

        See Also
        --------
        - [General Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/rectification/)
        - [Documentation page for Random Forest](https://www.cril.univ-artois.fr/pyxai/documentation/rectification/random_forest/)
        """
        

        return self.rectify_cxx(conditions=conditions, label=label, tests=tests)

        #
        # useless prototype implementation written in Python
        #
        current_time = time.process_time()
        
        Tools.verbose("")
        Tools.verbose("-------------- C++ Rectification information:")
        n_nodes_python = self._random_forest.n_nodes()
        Tools.verbose("Model - Number of nodes (initial):", n_nodes_python) 

        is_implicant = self.is_implicant(conditions, prediction=label)
        print("is_implicant ?", is_implicant)
        Tools.verbose("Label:", label)

        # Rectification part
        for i, tree in enumerate(self._random_forest.forest):

            tree_decision_rule = self._random_forest.forest[i].decision_rule_to_tree(conditions, label)
            if label == 1:
                # When label is 1, we have to inverse the decision rule and disjoint the two trees. 
                tree_decision_rule = tree_decision_rule.negating_tree() 
                self._random_forest.forest[i] = tree.disjoint_tree(tree_decision_rule)
            elif label == 0:
                # When label is 0, we have to concatenate the two trees.  
                self._random_forest.forest[i] = tree.concatenate_tree(tree_decision_rule)
            else:
                raise NotImplementedError("Multiclasses is in progress.")
            
        n_nodes_python = self._random_forest.n_nodes()
        Tools.verbose("Model - Number of nodes (after rectification):", n_nodes_python) 
        
        if tests is True:
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant after rectification ?", is_implicant)
            if is_implicant is False:
                raise ValueError("Problem 2")
            
        # Simplify Theory part
        for i, tree in enumerate(self._random_forest.forest):    
            self._random_forest.forest[i] = self.simplify_theory(tree)

        n_nodes_python = self._random_forest.n_nodes()
        Tools.verbose("Model - Number of nodes (after simplification using the theory):", n_nodes_python)
        
        if tests is True:
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant after simplify theory ?", is_implicant)
            if is_implicant is False:
                raise ValueError("Problem 3")

        # Simplify part
        for i, tree in enumerate(self._random_forest.forest):    
            tree.simplify()
        
        n_nodes_python = self._random_forest.n_nodes()
        Tools.verbose("Model - Number of nodes (after elimination of redundant nodes):", n_nodes_python)
        
        if tests is True:
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant after simplify ?", is_implicant)
            if is_implicant is False:
                raise ValueError("Problem 4")
            
        if self._instance is not None:
            self.set_instance(self._instance)

        self._elapsed_time = time.process_time() - current_time
        
        Tools.verbose("Rectification time:", self._elapsed_time)

        Tools.verbose("--------------")
        
        return self._random_forest