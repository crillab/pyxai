import numpy
import os
from math import floor
from pysat.card import CardEnc, EncType
from pyxai.sources.core.structure.treeEnsembles import TreeEnsembles
from pyxai.sources.core.structure.type import Encoding
from pyxai.sources.core.tools.encoding import CNFencoding


class RandomForest(TreeEnsembles):

    def __init__(self, forest, n_classes=2, learner_information=None):
        super().__init__(forest, learner_information)
        self.n_classes = n_classes
        self.learner_information = learner_information
        # assert all(tree.type_tree is TypeTree.PREDICTION for tree in self.forest), "All trees in a random forest have to be of the type PREDICTION."


    def raw_data(self):
        raw = tuple(tree.raw_data() for tree in self.forest)
        return (self.n_classes, raw)


    def predict_instance(self, instance):
        """
        Return the prediction (the classification) of an instance according to the trees
        """
        n_votes = numpy.zeros(self.n_classes)
        for tree in self.forest:
            n_votes[tree.predict_instance(instance)] += 1
        return numpy.argmax(n_votes)


    def __str__(self):
        s = "**Random Forest Model**" + os.linesep
        s += "nClasses: " + str(self.n_classes) + os.linesep
        s += "nTrees: " + str(self.n_trees) + os.linesep
        # s += "nFeatures in the biggest tree: " + str(max(tree.existing_variables() for tree in self.forest)) + os.linesep
        s += "nVariables: " + str(len(self.map_id_binaries_to_features) - 1) + os.linesep
        return s


    def is_implicant(self, implicant, prediction):
        forest_implicant = [tree.is_implicant(implicant, prediction) for tree in self.forest]
        n_trees = len(forest_implicant)
        n_trues = len([element for element in forest_implicant if element])
        return n_trues > int(n_trees / 2)


    def to_CNF(self, instance, binary_representation, target_prediction=None, *, tree_encoding=Encoding.COMPLEMENTARY,
               cardinality_encoding=Encoding.SEQUENTIAL_COUNTER):
        """Generate a CNF encoding two things:
           1) The Carsten-sinz coding of the cardinality constraint for n variables (with EncType.seqcounter)
           2) The trees of the forest (MethodCNF.COMPLEMENTARY)
        Args:
            instance (_type_): _description_
            target_prediction (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        n_original_variables = len(binary_representation)
        if target_prediction is None:
            target_prediction = self.take_decisions_instance(instance)

        cnf = []
        # We add firsly the cardinality constraint dealing with the votes of the trees in the forest.
        # Each tree is represented by a new variable
        new_variables_atleast = [v for v in range(1 + n_original_variables, 1 + n_original_variables + self.n_trees)]

        condition_atleast = floor(self.n_trees / 2) + 1
        cardinality_encoding = EncType.seqcounter if cardinality_encoding == Encoding.SEQUENTIAL_COUNTER else EncType.totalizer
        atleast_clauses = CardEnc.atleast(lits=new_variables_atleast, encoding=cardinality_encoding, bound=condition_atleast,
                                          top_id=new_variables_atleast[-1]).clauses

        # atleast_clauses = [[l + ((1 if l > 0 else -1)*n_original_variables) for l in clause] for clause in atleast_clauses]

        # new_variables_atleast = [l for l in range(1+n_original_variables,1+n_original_variables+self.n_trees)]

        cnf.extend(atleast_clauses)

        # We secondly encode the trees
        for i, new_variable in enumerate(new_variables_atleast):
            current_tree = self.forest[i]

            if tree_encoding == Encoding.COMPLEMENTARY:
                clauses_for_l = current_tree.to_CNF(instance, target_prediction, format=False)
                clauses_for_not_l = current_tree.to_CNF(instance, target_prediction, format=False, inverse_coding=True)
                for clause in clauses_for_l:
                    clause.append(-new_variable)
                for clause in clauses_for_not_l:
                    clause.append(new_variable)
                cnf.extend(clauses_for_l + clauses_for_not_l)
            elif tree_encoding == Encoding.SIMPLE:
                clauses_for_l = current_tree.to_CNF(instance, target_prediction, format=False)
                for clause in clauses_for_l:
                    clause.append(-new_variable)
                cnf.extend(clauses_for_l)
            elif tree_encoding == Encoding.MUS:
                clauses_for_l = current_tree.to_CNF(instance, target_prediction, format=False, inverse_coding=True)
                for clause in clauses_for_l:
                    clause.append(-new_variable)
                cnf.extend(clauses_for_l)
            else:
                assert False, "Bad parameter for " + str(tree_encoding) + " !"

        return CNFencoding.format(cnf)
