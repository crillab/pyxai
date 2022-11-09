import random
import time

import c_explainer
from pycsp3 import UNSAT, UNKNOWN

from pyxai.sources.core.explainer.Explainer import Explainer
from pyxai.sources.core.structure.type import ReasonExpressivity
from pyxai.sources.core.tools.utils import flatten
from pyxai.sources.solvers.CSP.AbductiveV1 import AbductiveModelV1
from pyxai.sources.solvers.CSP.TSMinimalV2 import TSMinimal
from pyxai.sources.solvers.GRAPH.TreeDecomposition import TreeDecomposition


class ExplainerBT(Explainer):

    def __init__(self, boosted_trees, instance=None):
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
        super().set_instance(instance)
        self._implicant_id_features = self._boosted_trees.get_id_features(self._binary_representation)


    def _to_binary_representation(self, instance):
        return self._boosted_trees.instance_to_binaries(instance)


    def is_implicant(self, abductive):
        if self._boosted_trees.n_classes == 2:
            # 2-classes case
            sum_weights = []
            for tree in self._boosted_trees.forest:
                weights = self.compute_weights(tree, tree.root, abductive)
                worst_weight = min(weights) if self.target_prediction == 1 else max(weights)
                sum_weights.append(worst_weight)

            sum_weights = sum(sum_weights)
            prediction = 1 if sum_weights > 0 else 0

            return self.target_prediction == prediction
        else:
            # multi-classes case
            worst_one = self.compute_weights_class(abductive, self.target_prediction, king="worst")
            best_ones = [self.compute_weights_class(abductive, cl, king="best") for cl
                         in self._boosted_trees.classes if cl != self.target_prediction]
            return all(worst_one > best_one for best_one in best_ones)


    def predict(self, instance):
        return self._boosted_trees.predict_instance(instance)


    def trees_statistics(self):
        # print("---------   Trees Information   ---------")
        n_nodes = sum([len(tree.get_variables()) for tree in self._boosted_trees.forest])
        n_nodes_biggest_tree = max([len(tree.get_variables()) for tree in self._boosted_trees.forest])
        n_nodes_biggest_tree_without_redundancy = max([len(set(tree.get_variables())) for tree in self._boosted_trees.forest])

        n_features = len(set(flatten([tree.get_features() for tree in self._boosted_trees.forest])))

        n_nodes_without_redundancy = []
        for tree in self._boosted_trees.forest:
            n_nodes_without_redundancy.extend(tree.get_variables())
        n_nodes_without_redundancy = len(list(set(n_nodes_without_redundancy)))

        return {"n_nodes": n_nodes,
                "n_nodes_without_redundancy": n_nodes_without_redundancy,
                "n_nodes_biggest_tree": n_nodes_biggest_tree,
                "n_nodes_biggest_tree_without_redundancy": n_nodes_biggest_tree_without_redundancy,
                "n_features": n_features}


    def reason_statistics(self, reason, *, reason_expressivity):
        if reason_expressivity == ReasonExpressivity.Conditions:
            if reason is not None:
                reason_length = len(reason)
                features_in_reason = self.to_features_indexes(reason)
                reduction_features = round(float(100 - (len(features_in_reason) * 100) / len(self._instance)), 2)
                reduction_conditions = round(float(100 - (len(reason) * 100) / len(self._binary_representation)), 2)
            else:
                reason_length = len(self._binary_representation)
                features_in_reason = []
                reduction_features = round(float(0), 2)
                reduction_conditions = round(float(0), 2)
            return {"instance_length": len(self._instance),
                    "implicant_length": len(self._binary_representation),
                    "reason_length": reason_length,
                    "features_in_reason_length": len(features_in_reason),
                    "features_not_in_reason_length": len(self._instance) - len(features_in_reason),
                    "%_reduction_conditions": reduction_conditions,
                    "%_reduction_features": reduction_features}
        elif reason_expressivity == ReasonExpressivity.Features:
            if reason is not None:
                reason_length = len(reason)
                reduction_features = round(float(100 - (len(reason) * 100) / len(self._instance)), 2)
            else:
                reason_length = 0
                reduction_features = round(float(0), 2)
            return {"instance_length": len(self._instance),
                    "reason_length": reason_length,
                    "features_in_reason_length": reason_length,
                    "features_not_in_reason_length": len(self._instance) - reason_length,
                    "%_reduction_features": reduction_features}
        else:
            assert True, "TODO"


    def to_features_indexes(self, reason):
        """Return the indexes of the instance that are involved in the reason.

        Args:
            reason (list): The reason.

        Returns:
            list: indexes of the instance that are involved in the reason.
        """
        features = [feature["id"] for feature in self.to_features(reason, details=True)]
        return [i for i, _ in enumerate(self._instance) if i + 1 in features]


    def to_features(self, binary_representation, *, eliminate_redundant_features=True, details=False):
        return self._boosted_trees.to_features(binary_representation, details=details)


    def redundancy_analysis(self):
        return self._boosted_trees.redundancy_analysis()


    def compute_propabilities(self):
        return self._boosted_trees.compute_probabilities(self._instance)


    def direct_reason(self):
        """The direct reason is the set of conditions used to classified the instance.

        Returns:
            list: The direct reason.
        """
        direct_reason = set()
        for tree in self._boosted_trees.forest:
            direct_reason |= set(tree.direct_reason(self._instance))

        # remove excluded features
        if any(not self._is_specific(lit) for lit in direct_reason):
            return None

        return Explainer.format(list(direct_reason))


    def sufficient_reason(self, *, n=1, seed=0, time_limit=None):
        """ Compute a sufficient reason using several CSP thanks to pycsp3 models.
        Works only on binary instances for the moment

        Returns:
            list: The sufficient reason.
        """
        assert n == 1, "To do implement that"
        if self._boosted_trees.n_classes > 2:
            raise NotImplementedError

        cp_solver = AbductiveModelV1()

        abductive = list(self._binary_representation).copy()
        is_removed = [False for _ in abductive]
        if seed != 0:
            random.seed(seed)
            random.shuffle(abductive)

        for i, lit in enumerate(abductive):
            is_removed[i] = True
            cp_solver.create_model_is_abductive(abductive, is_removed, self._boosted_trees, self.target_prediction)
            # cp_solver.create_model_is_abductive(abductive, i, is_removed, self._boosted_trees, self.target_prediction)
            result, solution = cp_solver.solve(time_limit=time_limit)
            if result == UNKNOWN:
                raise ValueError
            if result != UNSAT:  # We can not remove this literal because else at least one solution do not predict the good class !
                is_removed[i] = False

        abductive = [l for i, l in enumerate(abductive) if not is_removed[i]]

        return Explainer.format(abductive, n)


    def minimal_tree_specific_reason(self, *, time_limit=None, from_reason=None):
        cp_solver = TSMinimal()
        implicant_id_features = []  # TODO V2 self.implicant_id_features if reason_expressivity == ReasonExpressivity.Features else []
        cp_solver.create_model_minimal_abductive_BT(self._binary_representation, self._boosted_trees, self.target_prediction, self._boosted_trees.n_classes,
                                                    implicant_id_features, from_reason)
        time_used = -time.time()
        # TODO V2 reason_expressivity = reason_expressivity,
        tree_specific = self.tree_specific_reason(n_iterations=5)

        result, solution = cp_solver.solve(time_limit=time_limit, upper_bound=len(tree_specific) + 1)
        time_used += time.time()
        self._elapsed_time = time_used if result == "OPTIMUM" else Explainer.TIMEOUT

        return None if (result == UNSAT or result == UNKNOWN) else Explainer.format([l for i, l in enumerate(self._binary_representation) if solution[i] == 1])


    def tree_specific_reason(self, *, n_iterations=50, time_limit=None, seed=0):
        """
        Tree-specific (TS) explanations are abductive explanations that can be computed in polynomial time. While tree-specific explanations are not
        subset-minimal in the general case, they turn out to be close to sufficient reasons in practice. Furthermore, because sufficient reasons can
        be derived from tree-specific explanations, computing tree-specific explanations can be exploited as a preprocessing step in the derivation
        of sufficient reasons

        The method used (in c++), for a given seed, compute several tree specific reasons and return the best.
        For that, the algorithm is executed either during a given time or or until a certain number of reasons is calculated.

        The parameter 'reason_expressivity' have to be fixed either by ReasonExpressivity.Features or ReasonExpressivity.Conditions.

        Args:
            n_iterations (int, optional): _description_. Defaults to 50.
            time_limit (int, optional): _description_. Defaults to None.
            seed (int): _description_. the seed
        Returns:
            list: The tree-specific reason
        """
        # TODO V2,
        reason_expressivity = ReasonExpressivity.Conditions

        if seed is None:
            seed = -1
        if time_limit is None:
            time_limit = 0

        if self.c_BT is None:
            # Preprocessing to give all trees in the c++ library
            self.c_BT = c_explainer.new_BT(self._boosted_trees.n_classes)
            for tree in self._boosted_trees.forest:
                c_explainer.add_tree(self.c_BT, tree.raw_data_for_CPP())
        c_explainer.set_excluded(self.c_BT, tuple(self._excluded_literals))
        reason = c_explainer.compute_reason(self.c_BT, self._binary_representation, self._implicant_id_features, self.target_prediction, n_iterations, time_limit,
                                            int(reason_expressivity), seed)
        if reason_expressivity == ReasonExpressivity.Conditions:
            return reason
        elif reason_expressivity == ReasonExpressivity.Features:
            return self.to_features_indexes(reason)


    def compute_weights_class(self, implicant, cls, king="worst"):
        weights = [self.compute_weights(tree, tree.root, implicant) for tree in self._boosted_trees.forest if tree.target_class == cls]
        weights = [min(weights_per_tree) if king == "worst" else max(weights_per_tree) for weights_per_tree in weights]
        return sum(weights)


    @staticmethod
    def weight_float_to_int(weight):
        return weight
        # return int(weight*pow(10,9))


    def compute_weights(self, tree, node, implicant):

        if tree.root.is_leaf():  # Special case for tree without condition
            return [self.weight_float_to_int(tree.root.value)]

        id_variable = tree.get_id_variable(node)
        weights = []
        if id_variable in implicant:
            if node.right.is_leaf():
                return [self.weight_float_to_int(node.right.value)]
            else:
                weights.extend(self.compute_weights(tree, node.right, implicant))
                return weights
        elif -id_variable in implicant:
            if node.left.is_leaf():
                return [self.weight_float_to_int(node.left.value)]
            else:
                weights.extend(self.compute_weights(tree, node.left, implicant))
                return weights
        else:  # the variable is not in the implicant
            if node.left.is_leaf():
                weights.append(self.weight_float_to_int(node.left.value))
            else:
                weights.extend(self.compute_weights(tree, node.left, implicant))
            if node.right.is_leaf():
                weights.append(self.weight_float_to_int(node.right.value))
            else:
                weights.extend(self.compute_weights(tree, node.right, implicant))
        return weights


    def compute_tree_decomposition(self):
        """
        Compute the treewidth and the optimal tree decomposition.
        """
        tree_decomposition_solver = TreeDecomposition()
        tree_decomposition_solver.create_instance(self._boosted_trees)


    def is_tree_specific_reason(self, reason, check_minimal_inclusion=False):
        if not self.is_implicant(reason):
            return False
        if not check_minimal_inclusion:
            return True
        tmp = list(reason)
        random.shuffle(tmp)
        for lit in tmp:
            copy_reason = list(reason).copy()
            copy_reason.remove(lit)
            if self.is_implicant(tuple(copy_reason)):
                return False
        return True

# def check_sufficient(self, reason, n_samples=1000):
#   """
#   Check if the ''reason'' is abductive and check if the reasons with one selected literal in less are not abductives. This allows to check
#   approximately if the reason is sufficient or not.
#   Return nothing, just display the information.
#
#   Args:
#       reason (list): The reason.
#       n_samples (int, optional): Number of samples to test. Defaults to 1000.
#   """
#   percentage_abductive = self.check_abductive(reason, n_samples)
#   print("check_abductive:", percentage_abductive)
#   for lit in reason:
#     copy_reason = list(reason).copy()
#     copy_reason.remove(lit)
#     copy_reason = tuple(copy_reason)
#     percentage_current = self.check_abductive(copy_reason, n_samples)
#     print("check_sufficient:", percentage_current)
#
#
# def check_abductive(self, reason, n_samples=1000):
#   """
#   Check if ''n_samples'' of complete implicants created from a ''reason'' are alway of the same class. In other words, this method check
#   approximately if a reason seems abductive or not.
#
#   Args:
#       reason (list): The reason.
#       n_samples (int, optional): Number of samples to test. Defaults to 1000.
#
#   Returns:
#       float: The result is in the form of a percentage that represents the amount of complete implicants that are well classified.
#   """
#   ok_samples = 0
#   for _ in range(n_samples):
#     complete = self.extend_reason_to_complete_implicant(reason)
#     prediction = self._boosted_trees.is_implicant(complete)
#     if prediction == self.target_prediction:
#       ok_samples += 1
#   return round((ok_samples * 100) / n_samples, 2)
#
#
# def tree_specific_reason_python(self, seed=0):
#   """
#   Compute in python only one TS reason.
#   """
#   abductive = list(self.implicant).copy()
#   copy_implicant = list(self.implicant).copy()
#   if seed != 0:
#     random.seed(seed)
#     random.shuffle(copy_implicant)
#
#   for lit in copy_implicant:
#     tmp_abductive = abductive.copy()
#     tmp_abductive.remove(lit)
#     if self.is_implicant(tmp_abductive):
#       abductive.remove(lit)
#
#   return Explainer.format(abductive)
