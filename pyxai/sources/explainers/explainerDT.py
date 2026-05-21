import time

from pyxai.sources.explainers.Explainer import Explainer
from pyxai.sources.data_structures.decisionTree import DecisionTree
from pyxai.sources.type import PreferredReasonMethod, TypeTheory
from pyxai.sources.tools.encoding import CNFencoding
from pyxai.sources.tools.utils import compute_weight
from pyxai.sources.solvers.COMPILER.D4Solver import D4Solver
from pyxai.sources.solvers.MAXSAT.OPENWBOSolver import OPENWBOSolver
from pyxai.sources.solvers.SAT.glucoseSolver import GlucoseSolver
from pyxai import Tools

import c_explainer
import random
from pysat.solvers import Glucose3

###
# TODO: refactoring of preferred sufficient reason (keep only the method by weight and minimal)
# TODO: theory do not work for all explanation in DT 
# TODO:
###

class ExplainerDT(Explainer):
    r"""The Decision-tree explainer.

    This class adapts the ```Explainer``` class to a
    ```DecisionTree``` model. It provides methods to:

    - set an instance and derive its Boolean (literal-based) representation,
    - compute different families of explanations (direct, contrastive, sufficient, preferred),
    - query literals that are necessary or relevant for the current prediction,
    - optionally *rectify* (modify) the decision tree to enforce a desired label under given conditions.

    Some functions come from the parent class ```Explainer```, their documentation is available in the dedicated page:
    - Main Methods: ```set_instance```, ```set_features_type```, ```get_model```, ```predict```, ```activate_theory```, ```deactivate_theory```, ```set_excluded_features```, ```unset_excluded_features```;
    - Human-Readable Explanation Methods: ```to_features```, ```get_feature_names```, ```get_feature_names_from_literal```, ```reason_contains_features```;
    - Explanation verification methods: ```is_implicant```, ```is_reason```, ```is_sufficient_reason```, ```is_contrastive_reason```.

    See Also
    --------
    - The [documentation pages](https://www.cril.univ-artois.fr/pyxai/documentation/explanations/DTexplanations/)
    - The paper [On the Explanatory Power of Decision Trees](https://arxiv.org/abs/2108.05266)
    """

    def __init__(self, tree, instance=None):
        r"""Initialize an explainer for a decision tree.

        Parameters
        ----------
        tree : ```DecisionTree```
            The model in the form of a ```DecisionTree```.
        instance : ```list[float]``` (optional, default: None)
            The instance (observation) for which explanations will be computed.
            If ``None``, you must call ```set_instance``` before requesting explanations.
        """
        super().__init__()
        self._tree = tree  # The decision _tree.
        if instance is not None:
            self.set_instance(instance)
        self.c_rectifier = None


    @property
    def tree(self):
        return self._tree


    def set_instance(self, instance):
        r"""Set the instance to explain.

        Once the Explainer is built, you can explain several instances using it.
        Each time you want to explain a new instance, you need to call the function set_instance
        (it is not necessary to create a new Explainer object).

                Parameters
                ----------
                instance : ```list[float]```
                    The instance to explain.
                """
        super().set_instance(instance)
        self._n_sufficient_reasons = None


    def _to_binary_representation(self, instance):
        return self._tree.instance_to_binaries(instance)


    def is_implicant(self, binary_representation, *, prediction=None):
        r"""Check whether a set of literals entails a prediction for the tree.

        Parameters
        ----------
        binary_representation : ```list[int]```
            A (possibly partial) set of signed literals.
        prediction : ```int | None``` (optional, default: None)
            The prediction to test entailment for. The default value is the target prediction of the selected instance`.

        Returns
        -------
        bool:
            True iff the literals  force the given prediction, False otherwise.
        """
        if prediction is None:
            prediction = self.target_prediction
        binary_representation = self.extend_reason_with_theory(binary_representation)
        return self._tree._is_implicant(binary_representation, prediction)


    def predict(self, instance):
        r"""Predict the label/value for an instance using the underlying tree.

                Parameters
                ----------
                instance : ```list[float]```
                    Observation to classify/regress.

                Returns
                -------
                ```int```:
                    The tree prediction for the given instance.
                """
        return self._tree.predict_instance(instance)


    def to_features(self, binary_representation, *, eliminate_redundant_features=True, details=False, contrastive=False, without_intervals=False):
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
        binary_representation: ```list``` of ```int```
            the binary representation to convert

        eliminate_redundant_features: ```bool``` (optional, default: True)
            eliminate possible redundant features.

        details: ```bool``` (optional, default: False)
            returns a detailed representation instead of a string

        contrastive:  ```bool``` (optional, default: False):
            invert logic in order to explain a contrastive reason

        Returns
        -------
        ```list[str]```|```dict```:
            a list of string or of dictionaries depending the value of details.

        See Also
        --------
        - [Documentation page](http://localhost:4000/pyxai/documentation/explainer/concepts/)
        """
        return self._tree.to_features(binary_representation, details=details, eliminate_redundant_features=eliminate_redundant_features,
                                      contrastive=contrastive, without_intervals=without_intervals, feature_names=self.get_feature_names())

    #####################################################################################
    ### Methods for calculating explanations                                          ###
    #####################################################################################

    def direct_reason(self):
        r"""Compute the direct reason for the current instance.

        A *direct reason* corresponds to a subset of the binary representation describing the path taken by the
        instance in the tree (i.e., a complete assignment along the decision path). If the result
        contains excluded features, ``None`` is returned.

        Returns
        -------
        ```list[int]```:
            A list of literals representing the direct reason

        ```None```:
            If excluded features appear in the direct reason.

        Raises
        ------
        ValueError
            If no instance has been set.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/DTexplanations/direct/)
    """
        if self._instance is None:
            raise ValueError("Instance is not set")

        self._elapsed_time = 0
        direct_reason = self._tree.direct_reason(self._instance)
        if any(not self._is_specific(lit) for lit in direct_reason):
            direct_reason = None  # The reason contains excluded features
        else:
            direct_reason = Explainer.format(direct_reason)

        self._visualisation.add_history(self._instance, self.__class__.__name__, self.direct_reason.__name__, direct_reason)
        return direct_reason


    def contrastive_reason(self, *, n=1):
        r"""Compute up to ``n`` contrastive reasons for the current instance.

        Contrastive explanations explain why the instance has not been classified by the ML model as expected.
        Returns n contrastive reasons sorted by increasing of the size for the current instance in a Tuple
        (when n is set to 1, does not return a Tuple but just the reason). Supports excluded features.

        Parameters
        ----------
        n: ```int``` | ```Explaining.ALL```  (optional,  default=1)
            Maximum number of contrastive reasons to return. If a ```Explaining.ALL``` is provided,
            all computed candidates are formatted/returned.

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

        Note
        ----
        As the contrastive_reason returns the contrastive reasons in a ascending order according to their sizes,
        the minimal contrastive reasons are the first ones in the returned tuple.

        See Also
        --------
        - [Documentation page](http://www.cril.univ-artois.fr/pyxai/documentation/classification/DTexplanations/contrastive/)
        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        self._elapsed_time = 0
        cnf = self._tree.to_CNF(self._instance)
        core = CNFencoding.extract_core(cnf, self._binary_representation)
        core = [c for c in core if all(self._is_specific(lit) for lit in c)]  # remove excluded
        tmp = sorted(core, key=lambda clause: len(clause))
        if self._theory:  # Remove bad contrastive wrt theory
            contrastives = []
            for c in tmp:
                extended = self.extend_reason_with_theory([-lit for lit in c])
                if(len(extended) > 0):  # otherwise unsat => not valid with theory
                    contrastives.append(c)
        else:
            contrastives = tmp

        contrastives = Explainer.format(contrastives, n) if type(n) != int else Explainer.format(contrastives[:n], n)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.contrastive_reason.__name__, contrastives)
        return contrastives


    def necessary_literals(self):
        r"""Return literals that are necessary for the current prediction.

        Returns
        -------
        ```list[int]```
            Sorted list of necessary literals (may include excluded-feature literals).

        Raises
        ------
        ValueError
            If no instance has been set.
        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        self._elapsed_time = 0
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        core = CNFencoding.extract_core(cnf, self._binary_representation)
        # DO NOT remove excluded features. If they appear, they explain why there is no sufficient

        literals = sorted({lit for _, clause in enumerate(core) if len(clause) == 1 for lit in clause})
        #self.add_history(self._instance, self.__class__.__name__, self.necessary_literals.__name__, literals)
        return literals


    def relevant_literals(self):
        r"""Return literals that appear in non-unit clauses of the CNF core.

                Only literals corresponding to *specific* (non-excluded) features are kept.

                Returns
                -------
                ```list[int]```
                    list of relevant literals (order preserved by first appearance).

                Raises
                ------
                ValueError
                    If no instance has been set.
                """
        if self._instance is None:
            raise ValueError("Instance is not set")
        self._elapsed_time = 0
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        core = CNFencoding.extract_core(cnf, self._binary_representation)

        literals = [lit for _, clause in enumerate(core) if len(clause) > 1 for lit in clause if self._is_specific(lit)]  # remove excluded features
        #self.add_history(self._instance, self.__class__.__name__, self.relevant_literals.__name__, literals)
        return list(dict.fromkeys(literals))


    def _excluded_features_are_necesssary(self, prime_cnf):
        return any(not self._is_specific(lit) for lit in prime_cnf.necessary)

    def get_prime_implicant_CNF(self):
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        prime_implicant_cnf = CNFencoding.to_prime_implicant_CNF(cnf, self._binary_representation)
        return prime_implicant_cnf

    def sufficient_reason(self, *, n=1, time_limit=None):
        r"""Enumerate up to ```n``` sufficient reasons for the current prediction.

        A *sufficient reason* is a (typically) minimal subset of the binary representation that still entails the
        current prediction.

        Returns `n` sufficient reasons of the current instance in a Tuple (when n is set to 1, does not return
        a tuple but just the reason). Supports the excluded features. The reasons are in the form of binary variables,
        you must use the to_features method if you want to get a representation based on the features represented at start.

        Parameters
        ----------
        n: ```int``` | ```Explaining.ALL```, (optional, default=1)
            Maximum number of sufficient reasons to enumerate. If ```Explaining.ALL```, treated as unbounded.
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
            - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/DTexplanations/sufficient/)
        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        time_used = 0
        n = n if type(n) == int else float('inf')
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        prime_implicant_cnf = CNFencoding.to_prime_implicant_CNF(cnf, self._binary_representation)

        if self._excluded_features_are_necesssary(prime_implicant_cnf):
            self._elapsed_time = 0
            return []

        SATsolver = GlucoseSolver()
        SATsolver.add_clauses(prime_implicant_cnf.cnf)

        # Remove excluded features
        SATsolver.add_clauses([[-prime_implicant_cnf.mapping_original_to_new[abs(lit)]]
                               for lit in self._excluded_literals
                               if prime_implicant_cnf.mapping_original_to_new[abs(lit)] is not None])


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

        reasons = Explainer.format(sufficient_reasons, n)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__, reasons)
        return reasons


    def preferred_sufficient_reason(self, *, method, n=1, time_limit=None, weights=None, features_partition=None):
        r"""Compute sufficient reasons optimized by a preference criterion.

        This method searches for sufficient reasons that maximize a score derived from feature weights.
        Optimization is performed using a MaxSAT solver (hard clauses enforce correctness; soft clauses
        penalize selecting literals).

        Parameters
        ----------
        method : ```PreferredReasonMethod``` | ```str```
            The preference method used to discriminate reasons. 
            Possible values are defined in the ```PreferredReasonMethod``` enum.
        n : ```int``` | ```Explaining.ALL```, (optional, default=1)
            Maximum number of preferred reasons to return (ties with the identical best score are returned
            up to ``n``).
        time_limit: ```float``` (optional,  default=```None```)
            Time limit (seconds) for the whole enumeration. If reached, `elapsed_time` is set to ```Explaining.TIMEOUT```.
        weights : list[float] | None, optional
            per-feature weights
            If ```None```, weights are computed from using the parameter method.
        features_partition: ```object``` | ```None``` (optional, default: None)
            partitioning information passed to the weight computation helper.

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
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/DTexplanations/sufficient/)

        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        n = n if type(n) == int else float('inf')
        cnf = self._tree.to_CNF(self._instance)
        self._elapsed_time = 0

        prime_implicant_cnf = CNFencoding.to_prime_implicant_CNF(cnf, self._binary_representation)

        # excluded are necessary => no reason
        if self._excluded_features_are_necesssary(prime_implicant_cnf):
            return None

        cnf = prime_implicant_cnf.cnf
        if len(cnf) == 0:
            reasons = Explainer.format([[lit for lit in prime_implicant_cnf.necessary]], n=n)
            if method == PreferredReasonMethod.Minimal:
                self._visualisation.add_history(self._instance, self.__class__.__name__, self.minimal_sufficient_reason.__name__, reasons)
            else:
                self._visualisation.add_history(self._instance, self.__class__.__name__, self.preferred_sufficient_reason.__name__, reasons)
            return reasons

        weights = compute_weight(method, self._instance, weights, self._tree.learner_information, features_partition=features_partition)
        weights_per_feature = {i + 1: weight for i, weight in enumerate(weights)}

        soft = [lit for lit in prime_implicant_cnf.mapping_original_to_new if lit != 0]
        weights_soft = []
        for lit in soft:  # soft clause
            for i in range(len(self._instance)):
                #if self.to_features([lit], eliminate_redundant_features=False, details=True)[0]["id"] == i + 1:
                
                if self._tree.get_id_features([lit])[0] == i + 1:
                    weights_soft.append(weights[i])

        solver = OPENWBOSolver()

        # Hard clauses
        solver.add_hard_clauses(cnf)

        # Soft clauses
        for i in range(len(soft)):
            solver.add_soft_clause([-soft[i]], weights_soft[i])

        # Remove excluded features
        for lit in self._excluded_literals:
            if prime_implicant_cnf.from_original_to_new(lit) is not None:
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
            #score = sum([weights_per_feature[feature["id"]] for feature in
            #             self.to_features(preferred, eliminate_redundant_features=False, details=True)])
            
            score = sum([weights_per_feature[id_feature] for id_feature in self._tree.get_id_features(preferred)])
            if first_call:
                best_score = score
            elif score != best_score:
                break
            first_call = False
            reasons.append(preferred)
            if (time_limit is not None and time_used > time_limit) or len(reasons) == n:
                break
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        reasons = Explainer.format(reasons, n)
        if method == PreferredReasonMethod.Minimal:
            self._visualisation.add_history(self._instance, self.__class__.__name__, self.minimal_sufficient_reason.__name__, reasons)

        else:
            self._visualisation.add_history(self._instance, self.__class__.__name__, self.preferred_sufficient_reason.__name__, reasons)
        return reasons

    def minimal_sufficient_reason(self, *, n=1, time_limit=None):
        r"""Compute up to ```n``` sufficient reasons of minimal size.

        Parameters
        ----------
        n: ```int``` | ```Explaining.ALL```, (optional, default=1)
            Maximum number of sufficient reasons to enumerate. If ```Explaining.ALL```, treated as unbounded.
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

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/DTexplanations/sufficient/)
        """
        return self.preferred_sufficient_reason(method=PreferredReasonMethod.Minimal, n=n, time_limit=time_limit)


    def n_sufficient_reasons(self, time_limit=None):
        r"""Return the total number of sufficient reasons.

        Parameters
        ----------
        time_limit: ```float``` (optional,  default=```None```)
            Time limit (seconds) for the whole enumeration. If reached, `elapsed_time` is set to ```Explaining.TIMEOUT```.

        Returns
        -------
        ```int```:
            The total number of sufficient reasons.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/DTexplanations/sufficient/)

        """
        self.n_sufficient_reasons_per_attribute(time_limit=time_limit)
        return self._n_sufficient_reasons

    
    def n_sufficient_reasons_per_attribute(self, *, time_limit=None):
        r"""Return the total number of sufficient reasons per boolean attributes.

        Parameters
        ----------
        time_limit: ```float``` (optional,  default=```None```)
            Time limit (seconds) for the whole enumeration. If reached, `elapsed_time` is set to ```Explaining.TIMEOUT```.

        Returns
        -------
        ```dict```:
            A dictionary where the keys are the literals and the values the numbers of sufficient reasons.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/DTexplanations/sufficient/)

        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        cnf = self._tree.to_CNF(self._instance)
        prime_implicant_cnf = CNFencoding.to_prime_implicant_CNF(cnf, self._binary_representation)

        if self._excluded_features_are_necesssary(prime_implicant_cnf):
            self._elapsed_time = 0
            self._n_sufficient_reasons = 0
            return None

        if len(prime_implicant_cnf.cnf) == 0:  # Special case where all in necessary
            return {lit: 1 for lit in prime_implicant_cnf.necessary}

        compiler = D4Solver()
        # Remove excluded features
        cnf = list(prime_implicant_cnf.cnf)
        for lit in self._excluded_literals:
            if prime_implicant_cnf.from_original_to_new(lit) is not None:
                cnf.append([-prime_implicant_cnf.from_original_to_new(lit)])

        compiler.add_cnf(cnf, prime_implicant_cnf.n_literals - 1)
        compiler.add_count_model_query(cnf, prime_implicant_cnf.n_literals - 1, prime_implicant_cnf.n_literals_mapping)

        time_used = -time.time()
        n_models = compiler.solve(time_limit)
        
        self._elapsed_time = Explainer.TIMEOUT if n_models[1] == -1 else time_used
        if self._elapsed_time == Explainer.TIMEOUT:
            self._n_sufficient_reasons = None
            return {}

        self._n_sufficient_reasons = n_models[0]
        time_used += time.time()
        n_necessary = n_models[0] if len(n_models) > 0 else 1

        n_sufficients_per_attribute = {n: n_necessary for n in prime_implicant_cnf.necessary}
        for lit in range(1, prime_implicant_cnf.n_literals_mapping):
            n_sufficients_per_attribute[prime_implicant_cnf.mapping_new_to_original[lit]] = n_models[lit]

        return n_sufficients_per_attribute


    def is_reason(self, reason, *, n_samples=-1):
        r"""Return if the reason given in parameter is really a reason. Since the process can be time consuming, one limits the
        number of checks.

        Parameters
        ----------
        reason: (```list``` of ```float```)
            The reason to be tested.
        n_samples: (```int```, optional, default: 1000) the number of tests to be done.

        Returns
        -------
            True if the reason is really one reason, False otherwise.
        """
        return self._tree._is_implicant(reason, self.target_prediction)


    def rectify_cxx(self, *, conditions, label, tests=False):
        r"""Rectify the decision tree using the C++ backend.

        Rectification enforces that the (parsed) ``conditions`` imply the given ``label`` in the model,
        then simplifies the resulting tree (optionally leveraging the explainer's theory constraints).

        Parameters
        ----------
        conditions: ```list``` of ```int```
            Conditions (premises part of the classification rule) in the form of a binary representation (list of literals).
        label : ```int```
            Label representing the conclusion part of the classification rule.
        tests : ```bool``` (optional, default: False)
            If True, runs additional post-steps checks (can be expensive) to ensure rectification
            preserved the intended implicant property.

        Returns
        -------
        ```DecisionTree```
            The rectified (and simplified) decision tree.

        Raises
        ------
        ValueError
            If internal validation fails when ``tests=True``.

        See Also
        --------
        - [General Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/rectification/)
        - [Documentation page for decision Tree](https://www.cril.univ-artois.fr/pyxai/documentation/rectification/decision_tree/)
        """

        #check conditions and return a list of literals
        
        conditions, change = self._tree.parse_conditions_for_rectify(conditions)
        if change is True and self._last_features_types is not None:
            self.set_features_type(self._last_features_types)
       
        current_time = time.process_time()
        if self.c_rectifier is None:
            self.c_rectifier = c_explainer.new_rectifier()

        #if tests is True: # TODO remove 13/02/2026
        #    is_implicant = self.is_implicant(conditions, prediction=label)
        #    print("is_implicant ?", is_implicant)
        
        c_explainer.rectifier_add_tree(self.c_rectifier, self._tree.raw_data_for_CPP())
        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - Initial (c++):", n_nodes_cxx) 

        # Rectification part
        c_explainer.rectifier_improved_rectification(self.c_rectifier, conditions, label)
        n_nodes_ccx =  c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After rectification (c++):", n_nodes_ccx)    
        if tests is True:
            
            #for i in range(len(self._random_forest.forest)):
            tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, 0)
            self._tree.delete(self._tree._root)
            self._tree._root = self._tree.from_tuples(tree_tuples)
            is_implicant = self.is_implicant(conditions, prediction=label)
            if is_implicant is False:
                raise ValueError("Problem: the condition is not an imlicant of the prediction after rectification!")
        

        # Simplify Theory part
        theory_cnf = self.get_model().get_theory(None)
        c_explainer.rectifier_set_theory(self.c_rectifier, tuple(theory_cnf))
        c_explainer.rectifier_simplify_theory(self.c_rectifier)

        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After simplification with the theory (c++):", n_nodes_cxx)

        if tests is True: 
            tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, 0)
            self._tree.delete(self._tree._root)
            self._tree._root = self._tree.from_tuples(tree_tuples)
            is_implicant = self.is_implicant(conditions, prediction=label)
            if is_implicant is False:
                raise ValueError("Problem: the condition is not an imlicant of the prediction after simplification with the theory!")
        
        # Simplify part
        c_explainer.rectifier_simplify_redundant(self.c_rectifier)
        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After elimination of redundant nodes (c++):", n_nodes_cxx)
        
        # Get the C++ trees and convert it :) 
        tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, 0)
        self._tree.delete(self._tree._root)
        self._tree._root = self._tree.from_tuples(tree_tuples)
        
        
        c_explainer.rectifier_free(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - Final (c++):", self._tree.n_nodes())
        if tests is True:
            is_implicant = self.is_implicant(conditions, prediction=label)
            if is_implicant is False:
                raise ValueError("Problem: the decision rule is not an implicant of the tree after simplification!")
        
        if self._instance is not None:
            self.set_instance(self._instance)

        self._elapsed_time = time.process_time() - current_time
        
        Tools.verbose("Rectification time:", self._elapsed_time)

        Tools.verbose("--------------")
        return self._tree

    def rectify(self, *, conditions, label, tests=False):
        r"""Rectify the decision tree to satisfy a labeled constraint.

        Rectification enforces that the (parsed) ``conditions`` imply the given ``label`` in the model,
        then simplifies the resulting tree (optionally leveraging the explainer's theory constraints).




        Parameters
        ----------
        conditions: ```list``` of ```int```
            Conditions (premises part of the classification rule) in the form of a binary representation (list of literals).
        label : ```int```
            Label representing the conclusion part of the classification rule.
        tests: ```bool``` (optional, default: False)
            If True, runs additional post-steps checks (can be expensive) to ensure rectification
            preserved the intended implicant property.

        Returns
        -------
        ```DecisionTree```
            The rectified (and simplified) decision tree.

        Raises
        ------
        ValueError
            If internal validation fails when ``tests=True``.

        See Also
        --------
        - [General Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/rectification/)
        - [Documentation page for decision Tree](https://www.cril.univ-artois.fr/pyxai/documentation/rectification/decision_tree/)
        """

        return self.rectify_cxx(conditions=conditions, label=label, tests=tests)

        # useless prototype implementation written in Python
        Tools.verbose("")
        Tools.verbose("-------------- Rectification information:")

        is_implicant = self._tree._is_implicant(conditions, label)
        print("is_implicant before rectification ?", is_implicant)

        tree_decision_rule = self._tree.decision_rule_to_tree(conditions, label)
        Tools.verbose("Classification Rule - Number of nodes:", tree_decision_rule.n_nodes())
        Tools.verbose("Model - Number of nodes:", self._tree.n_nodes())
        if label == 1:
            # When label is 1, we have to inverse the decision rule and disjoint the two trees.  
            tree_decision_rule = tree_decision_rule.negating_tree()
            tree_rectified = self._tree.disjoint_tree(tree_decision_rule)
        elif label == 0:
            # When label is 0, we have to concatenate the two trees.  
            tree_rectified = self._tree.concatenate_tree(tree_decision_rule)
        else:
            raise NotImplementedError("Multiclasses is in progress.")
        
        print("tree_rectified:", tree_rectified.raw_data_for_CPP())
        print("label:", label)

        is_implicant = tree_rectified._is_implicant(conditions, label)
        print("is_implicant after rectification ?", is_implicant)
        if is_implicant is False:
            raise ValueError("Problem 2")
        
        Tools.verbose("Model - Number of nodes (after rectification):", tree_rectified.n_nodes())  
        tree_rectified = self.simplify_theory(tree_rectified)

        is_implicant = tree_rectified._is_implicant(conditions, label)
        print("is_implicant after rectification ?", is_implicant)
        if is_implicant is False:
            raise ValueError("Problem 3")
        
        Tools.verbose("Model - Number of nodes (after simplification using the theory):", tree_rectified.n_nodes())
        tree_rectified.simplify()
        Tools.verbose("Model - Number of nodes (after elimination of redundant nodes):", tree_rectified.n_nodes())
    
        self._tree = tree_rectified
        if self._instance is not None:
            self.set_instance(self._instance)
        Tools.verbose("--------------")
        return self._tree
        


    @staticmethod
    def _rectify_tree(_tree, positive_rectifying__tree, negative_rectifying__tree):
        not_positive_rectifying__tree = positive_rectifying__tree.negating_tree()
        not_negative_rectifying__tree = negative_rectifying__tree.negating_tree()

        _tree_1 = positive_rectifying__tree.concatenate_tree(not_negative_rectifying__tree)
        _tree_2 = negative_rectifying__tree.concatenate_tree(not_positive_rectifying__tree)

        not__tree_2 = _tree_2.negating_tree()

        _tree_and_not__tree_2 = _tree.concatenate_tree(not__tree_2)
        _tree_and_not__tree_2.simplify()

        _tree_and_not__tree_2_or__tree_1 = _tree_and_not__tree_2.disjoint_tree(_tree_1)

        _tree_and_not__tree_2_or__tree_1.simplify()
        
        return _tree_and_not__tree_2_or__tree_1

    def anchored_reason(self, *, n_anchors=2, reference_instances, time_limit=None, check=False):
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction, inverse_coding=True)
        n_variables = CNFencoding.compute_n_variables(cnf)
        
        return self._most_anchored_reason_anchored_reason(n_variables=n_variables, cnf=cnf, n_anchors=n_anchors, reference_instances=reference_instances, time_limit=time_limit, check=check)

