import random
import time

import c_explainer
from pycsp3 import UNSAT, UNKNOWN

from pyxai.sources.explainers.Explainer import Explainer
from pyxai.sources.type import ReasonExpressivity
from pyxai.sources.tools.utils import flatten
from pyxai.sources.solvers.CSP.AbductiveV1 import AbductiveModelV1
from pyxai.sources.solvers.CSP.TSMinimalV2 import TSMinimal
from pyxai.sources.solvers.GRAPH.TreeDecomposition import TreeDecomposition
from pyxai.sources.solvers.MIP.ContrastiveBT import ContrastiveBT
from pyxai.sources.solvers.ORTOOLS.IsImplicantBT import IsImplicantBT

import numpy

###
# Version 2.0 refactoring: 2024-06-10
# TODO: minimal tree specific reasons for BTs (multiclass case)
# TODO: for tree_specific_reason: what is theta ? 
###

class ExplainerBT(Explainer):
    r"""The Boosted Trees explainer.

    This class adapts the generic :class:```Explainer``` to a
    ```BoostedTrees``` model. It provides methods to:

    - set an instance and derive its Boolean (literal-based) representation,
    - compute different families of explanations (direct, contrastive, sufficient, tree_specific),
    - optionally *rectify* (modify) the random forest to enforce a desired label under given conditions.

    Some functions come from the parent class ```Explainer```, their documentation is available in the dedicated page:
    - Main Methods: ```set_instance```, ```set_features_type```, ```get_model```, ```predict```, ```activate_theory```, ```deactivate_theory```, ```set_excluded_features```, ```unset_excluded_features```;
    - Human-Readable Explanation Methods: ```to_features```, ```get_feature_names```, ```get_feature_names_from_literal```, ```reason_contains_features```;
    - Explanation verification methods: ```is_implicant```, ```is_reason```, ```is_sufficient_reason```, ```is_contrastive_reason```.

    See Also
    --------
    - The [documentation pages](https://www.cril.univ-artois.fr/pyxai/documentation/explanations/BTexplanations/)
    - The paper [Computing Abductive Explanations for Boosted Trees](https://arxiv.org/abs/2209.07740)

    Notes
    -----
    Some methods are limited to binary classification (2 classes). When multi-class support
    is not available, an exception `NotImplementedError` is raised.
    """

    #####################################################################################
    ### Base Methods                                                                  ###
    #####################################################################################

    def __init__(self, boosted_trees, instance=None):
        """
        Initialize an explainer for a Boosted Tree model.

        Parameters
        ----------
        boosted_trees: ```BoostedTrees```
            The Boosted Trees model to explain.
        instance : ```list[int]``` (optional, default: None)
            The instance (observation) for which explanations will be computed.
            If ``None``, you must call set_instance before requesting explanations.
        """
        super().__init__()
        self._boosted_trees = boosted_trees  # The boosted trees.
        self._implicant_id_features = []
        self.c_BT = None
        if instance is not None:
            self.set_instance(instance)

    @property
    def boosted_trees(self):
        return self._boosted_trees

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
        self._implicant_id_features = self._boosted_trees.get_id_features(self._binary_representation)

    def _to_binary_representation(self, instance):
        return self._boosted_trees.instance_to_binaries(instance)

    def _theory_clauses(self):
        return self._boosted_trees.get_theory(self._binary_representation)
    
    def predict(self, instance):
        r"""Predict the label/value for an instance using the underlying tree.

        Parameters
        ----------
        instance: ```list[float]```
            Observation to classify/regress.

        Returns
        -------
        ```int```:
            The prediction for the given instance.
        """

        return self._boosted_trees.predict_instance(instance)

    def to_features(self, binary_representation, *, eliminate_redundant_features=True, details=False, contrastive=False,
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
                ```list``` of ```str```|```dict```:
                    a list of string or of dictionaries depending the value of details.

                See Also
                --------
                - [Documentation page](http://localhost:4000/pyxai/documentation/explainer/concepts/)
        """

        return self._boosted_trees.to_features(binary_representation,
                                               eliminate_redundant_features=eliminate_redundant_features,
                                               details=details,
                                               contrastive=contrastive, without_intervals=without_intervals,
                                               feature_names=self.get_feature_names())

    #####################################################################################
    ### Methods related to the checking of the implicant and the tree specific reason ###
    #####################################################################################
    
    def is_reason(self, reason, *, n_samples=1000):
        raise NotImplementedError("The method is_reason is not implemented for boosted trees. Please use the method is_tree_specific_reason instead.")

    def is_implicant(self, binary_representation):
        """
        Check whether a ```binary_representation``` (partial or complete) implies the prediction. 
        This method uses a MIP (Mixed Integer Programming) model to check whether the prediction is the same for all complete implicants that can be created from the ```binary_representation ``` given as parameter. 
        Therefore, this method is exact and not heuristic. However, it can be time-consuming for large instances.

        Warning
        -------
        The checking process is complete and can be time-consuming. 

        Parameters
        ----------
        binary_representation: (```list[int]```)
            An implicant (partial or complete).

        Returns
        ```bool```
            True if the implicant is an implicant of the selected instance, False otherwise.
        """
        starting_time = -time.process_time()
        base_score = self._boosted_trees.learner_information.extras["base_score"]
        binary_instance = list(self._binary_representation).copy()
        theory = tuple(self._boosted_trees.get_theory(self._binary_representation)) if self._theory else None
        model = IsImplicantBT(self._boosted_trees, binary_instance, binary_representation, self.target_prediction, theory, base_score)
        
        time_used = starting_time + time.process_time()
        self._elapsed_time = time_used

        return model.is_implicant()

    def __is_tree_specific_reason(self, binary_representation):
        """
        Check whether a binary representation (partial or complete) respect the condition of tree specific reason. 
        That is to say, check if sum of maximal (or minimal depending on the target prediction) weights of the trees respect is less than or greater than 0 (according to the target prediction). 
        
        Parameters
        ----------
        binary_representation: (```list[int]```)
            the binary representation

        Returns
        ```bool```
            True if the binary representation is an implicant of the selected instance, False otherwise.
        """
        if self._boosted_trees.learner_information.problem_type != 'classification':
            raise NotImplementedError("The method is_tree_specific_reason is only implemented for classification boosted trees.")
        
        if self._boosted_trees.n_classes == 2:
            # 2-classes case
            base_score = self._boosted_trees.learner_information.extras["base_score"]
            sum_weights = []
            for tree in self._boosted_trees.forest:
                weights = self.compute_weights(tree, tree._root, binary_representation)
                worst_weight = min(weights) if self.target_prediction == 1 else max(weights)
                sum_weights.append(worst_weight)
            sum_weights = sum(sum_weights)
            base_score = -numpy.log(1 / base_score - 1) if base_score != 0 else 0
            prediction = 1 if sum_weights + base_score > 0 else 0
            return self.target_prediction == prediction
        else:

            # multi-classes case
            # TODO: I don't kwow how to take into account the base_score in the multi-class case, I will assume that it is the same for all classes and that it is added to the sum of the weights of each class. 
            worst_one = self.compute_weights_class(binary_representation, self.target_prediction, king="worst")
            best_ones = [self.compute_weights_class(binary_representation, cl, king="best") for cl
                         in self._boosted_trees.class_names if cl != self.target_prediction]
            return all(worst_one > best_one for best_one in best_ones)
    
    def is_tree_specific_reason(self, reason, check_minimal_inclusion=False):
        r"""Heuristically check whether a given reason (subset of the binary representation) behaves like a tree specific reason.

        Parameters
        ----------
        reason: ```list[int]```
            Candidate reason (literals).
        check_minimal_inclusion: ```bool``` (optional, default: False)
            Check if it is minimal for the subset operator (set theory). That is to say,
            there is no tree specific reason that is a strict subset of the reason given as parameter.
        
        Returns
        -------
        ```bool```:
            True if the reason passes the implicant and (heuristic) minimality checks, False otherwise.
        """

        extended_reason = self.extend_reason_with_theory(reason)
        if not self.__is_tree_specific_reason(extended_reason):
            return False

        if not check_minimal_inclusion:
            return True

        tmp = list(reason)
        random.shuffle(tmp)
        for lit in tmp:
            copy_reason = list(reason).copy()
            copy_reason.remove(lit)
            if self.__is_tree_specific_reason(tuple(copy_reason)):
                return False
        return True

    #####################################################################################
    ### Explanation Methods                                                           ###
    #####################################################################################

    def direct_reason(self):
        r"""Compute the direct reason for the current instance.

        The direct reason is obtained by collecting the root-to-leaf path conditions
        that cover the instance across the forest, If the result contains excluded features, ``None`` is returned.

        Returns
        -------
        ```list[int]```:
            A list of literals representing the direct reason
        ```None```:
            if excluded features appear.

        Raises
        ------
        ValueError
            If no instance has been set.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/BTexplanations/direct/)
        """

        if self._instance is None:
            raise ValueError("Instance is not set")

        direct_reason = set()
        for tree in self._boosted_trees.forest:
            direct_reason |= set(tree.direct_reason(self._instance))

        # remove excluded features
        if any(not self._is_specific(lit) for lit in direct_reason):
            return None

        reason = Explainer.format(list(direct_reason))
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.direct_reason.__name__, reason)
        return reason

    def tree_specific_reason(self, *, n_iterations=50, time_limit=None, seed=0, history=True, theta=0):
        r"""Compute a tree specific reason for the current instance.

        This method calls a greedy algorithm to compute a tree-specific explanation (that are, by definition, subset minimal for the inclusion). 
        The algorithm is run ```n_iterations``` times and takes the smallest (according to the size of the reason) tree specific reason that has been computed is returned. 
        
        Excluded features are supported. 
        

        Warning
        -------
        The method is heuristic and does not guarantee to return a minimal tree specific reason. 
        However, it is much faster than the exact method but it returns only one tree specific reason.
        
        Parameters
        ----------
        n_iterations: ```int``` (optional, default: ```50```)
            Number of randomized iterations used to extract the smallest one.
        time_limit: ```float``` (optional,  default=```None```)
            Time limit (seconds) for the whole enumeration. If reached, `elapsed_time` is set to ```Explaining.TIMEOUT```.
        seed: ```int``` (optional, default: ```0```)
            Random seed for the randomized iterations. Set to ```0``` to have a random seed.

        Returns
        -------
        ```tuple[int]```
            The tree specific reason
         ```None```:
            if computed reasons contain excluded features.

        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/BTexplanations/treespecific/)

        Raises
        ------
        ValueError:
            If no instance has been set.
        """
        if self._instance is None:
            raise ValueError("Instance is not set")

        if seed is None:
            seed = -1
        if time_limit is None:
            time_limit = 0

        if self.c_BT is None:
            # Preprocessing to give all trees in the c++ library
            self.c_BT = c_explainer.new_classifier_BT(self._boosted_trees.n_classes)

            for tree in self._boosted_trees.forest:
                c_explainer.add_tree(self.c_BT, tree.raw_data_for_CPP())

        c_explainer.set_excluded(self.c_BT, tuple(self._excluded_literals))
        if self._theory:
            c_explainer.set_theory(self.c_BT, tuple(self._boosted_trees.get_theory(self._binary_representation)))

        c_explainer.set_base_score(self.c_BT, self._boosted_trees.learner_information.extras["base_score"])

        reason = c_explainer.compute_reason(self.c_BT, self._binary_representation, self._implicant_id_features,
                                            self.target_prediction, n_iterations,
                                            time_limit,
                                            ReasonExpressivity.Conditions, seed, theta)
        
        reason = Explainer.format(reason)
        if history:
            self._visualisation.add_history(self._instance, self.__class__.__name__, self.tree_specific_reason.__name__,
                                            reason)
        return reason
    
    
    def minimal_tree_specific_reasons(self, *, n=10, time_limit=None):
        r"""Compute ```n``` minimal tree specific reasons for the current instance.

        This method considers an Mixed Integer Programming problem (MIP) representing the boosted tree encoded. 
        To solve it, several calls to a solver are performed and the result of each call is a minimal tree-specific explanation. 
        The method prevents from finding the same explanation twice or more by adding constraints between each invocation.
        
        Excluded features are not supported. 
        
        The reasons are in the form of binary variables, you must use the ```to_features``` method  if  you want to obtain a representation based on the features considered at start.

        Warning
        -------
        This method is complete and therefore can be time-consuming. 

        Parameters
        ----------
        n: ```int``` (default: ```10```)
            Number of minimal tree-pecific explanations to compute. 
        time_limit: ```float``` (optional,  default=```None```)
            Time limit (seconds) for the whole enumeration. If reached, `elapsed_time` is set to ```Explaining.TIMEOUT```.

        
        Returns
        -------
        ```tuple[int]``` | ```None```
            Returns n minimal tree-pecific explanations of the current instance in a Tuple (when n is set to 1, does not return a Tuple but just the reason). 
        
        See Also
        --------
        - [Documentation page](https://www.cril.univ-artois.fr/pyxai/documentation/classification/BTexplanations/treespecific/)

        Raises
        ------
        ValueError:
            If no instance has been set.
        """
        
        raise NotImplementedError("minimal_tree_specific_reasons is not available in V2 and will be re-introduced in V3.")
    
    
    def minimal_contrastive_reason(self, *, n=1, time_limit=None):
        r"""Compute ```n``` minimal contrastive reasons for the current instance.

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
        ValueError:
            If no instance has been set.

        See Also
        --------
        - [Documentation page](http://www.cril.univ-artois.fr/pyxai/documentation/classification/BTexplanations/contrastive/)
        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        if self._boosted_trees.n_classes > 2:
            raise NotImplementedError("Minimal contrastive reason is not implemented for the multi class case")

        starting_time = -time.process_time()
        contrastive_bt = ContrastiveBT()
        c = contrastive_bt.create_model_and_solve(self, None if self._theory is False else self._theory_clauses(),
                                                  self._excluded_literals, n, time_limit)
        time_used = starting_time + time.process_time()
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        self._visualisation.add_history(self._instance, self.__class__.__name__,
                                        self.minimal_contrastive_reason.__name__, c)
        return c
    

    #####################################################################################
    ### Util Methods                                                                  ###
    #####################################################################################
    def compute_weights_class(self, implicant, cls, king="worst"):
        """
        Used in is_tree_specific_reason to compute the sum of maximal (or minimal depending on the target prediction) weights of the trees for a given class.
        """
        weights = [self.compute_weights(tree, tree._root, implicant) for tree in self._boosted_trees.forest if
                   tree._associated_class == cls]
        weights = [min(weights_per_tree) if king == "worst" else max(weights_per_tree) for weights_per_tree in weights]
        return sum(weights)

    def compute_weights(self, tree, node, implicant):
        """
        Used in compute_weights_class to compute the weights of the trees for a given implicant.
        """
        if tree._root.is_leaf():  # Special case for tree without condition
            return [tree._root.value]

        id_variable = tree.get_id_variable(node)
        weights = []
        if id_variable in implicant:
            if node.right.is_leaf():
                return [node.right.value]
            else:
                weights.extend(self.compute_weights(tree, node.right, implicant))
                return weights
        elif -id_variable in implicant:
            if node.left.is_leaf():
                return [node.left.value]
            else:
                weights.extend(self.compute_weights(tree, node.left, implicant))
                return weights
        else:  # the variable is not in the implicant
            if node.left.is_leaf():
                weights.append(node.left.value)
            else:
                weights.extend(self.compute_weights(tree, node.left, implicant))
            if node.right.is_leaf():
                weights.append(node.right.value)
            else:
                weights.extend(self.compute_weights(tree, node.right, implicant))
        return weights

    
