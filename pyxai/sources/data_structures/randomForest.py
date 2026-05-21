import numpy
import os
from math import floor
from pysat.card import CardEnc, EncType
from pyxai.sources.data_structures.binaryMapping import BinaryMapping
from pyxai.sources.data_structures.treeEnsembles import TreeEnsembles
from pyxai.sources.type import Encoding
from pyxai.sources.tools.encoding import CNFencoding
from pyxai.sources.data_structures.baseModel import BaseModel
from pyxai.sources.learners.learner_information import LearnerInformation


class RandomForest(TreeEnsembles, BaseModel):

    def __init__(self, forest, n_features=None, n_classes=None, learner_information=None, feature_names=None, class_names=None):
        """
        Returns a ```RandomForest``` consisting of trees (the ```forest```) and ```n_classes``` to predict.

        Parameters
        ----------
        forest : ```list``` of ```DecisionTree```
            The list of trees composing the random forest.  
        n_classes : ```int```
            The number of classes to predict (can be multi-class).  
        learner_information : ```LearnerInformation``` (optional, default=LearnerInformation(problem_type='classification') if called from Builder else None)
            The information about the learning process.  
        feature_names : ```list``` of ```str``` (optional, default=None)
            The names of the features used in the model.  
        class_names : ```list``` of ```str``` (optional, default=None)
            The names of the classes to predict.
       
        Returns
        -------
        ```RandomForest```:
            The random forest model.

        Examples
        --------
        >>> node_t1_v1_1 = Builder.DecisionNode(1, operator=Builder.GE, threshold=10, left=0, right=0)
        >>> node_t1_v1_2 = Builder.DecisionNode(1, operator=Builder.GE, threshold=20, left=node_t1_v1_1, right=0)
        >>> node_t1_v1_3 = Builder.DecisionNode(1, operator=Builder.GE, threshold=30, left=node_t1_v1_2, right=1)
        >>> node_t1_v1_4 = Builder.DecisionNode(1, operator=Builder.GE, threshold=40, left=node_t1_v1_3, right=1)
        >>> tree_1 = Builder.DecisionTree(3, node_t1_v1_4)

        >>> node_t2_v3 = Builder.DecisionNode(3, operator=Builder.EQ, threshold=1, left=0, right=1)
        >>> node_t2_v2 = Builder.DecisionNode(2, operator=Builder.EQ, threshold=1, left=0, right=node_t2_v3)
        >>> tree_2 = Builder.DecisionTree(3, node_t2_v2)

        >>> tree_3 = Builder.DecisionTree(3, Builder.LeafNode(1))
        >>> forest = Builder.RandomForest([tree_1, tree_2, tree_3], n_classes=2)
        """
        BaseModel.__init__(self, n_features, n_classes, learner_information, feature_names, class_names)
        TreeEnsembles.__init__(self, forest)        
        # assert all(tree.type_tree is TypeTree.PREDICTION for tree in self.forest), "All trees in a random forest have to be of the type PREDICTION."  Not used 19/09/2025


    def raw_data(self):
        raw = tuple(tree.raw_data() for tree in self.forest)
        return (self.n_classes, raw)

    
    def _predict_votes(self, instance):
        """
        Return the number of votes for each class
        """
        n_votes = numpy.zeros(self.n_classes)
        for tree in self.forest:
            n_votes[tree._predict_instance(instance)] += 1
        return n_votes, numpy.argmax(n_votes)

    
    def _predict_instance(self, instance):
        """
        Internal function for prediction computation. Not for direct use
        Return the prediction (the classification) of an instance according to the trees
        """
        if self.n_classes is None:
            raise ValueError("The number of classes in the random forest is unknown")
        n_votes = numpy.zeros(self.n_classes)
        for tree in self.forest:
            n_votes[tree._predict_instance(instance)] += 1
        return numpy.argmax(n_votes)

    
    def _predict_implicant(self, implicant):
        """
        Return the prediction (the classification) of an instance according to the trees
        """
        n_votes = numpy.zeros(self.n_classes)
        for tree in self.forest:
            n_votes[tree._take_decisions_binary_representation(implicant, self.map_features_to_id_binaries)] += 1
        return numpy.argmax(n_votes)
        
        
    def __str__(self):
        s = "**Random Forest Model**" + os.linesep
        s += "nClasses: " + str(self.n_classes) + os.linesep
        s += "nTrees: " + str(self.n_trees) + os.linesep
        s += "nVariables: " + str(len(self.map_id_binaries_to_features) - 1) + os.linesep
        return s


    ###
    # Do not use directly, as the implicant cannot coincide with the theory 
    # (use is_implicant of explainer object instead)
    ###
    def _is_implicant(self, implicant, prediction):
        if self.n_classes == 2:
            forest_implicant = [tree._is_implicant(implicant, prediction) for tree in self.forest]
            n_trees = len(forest_implicant)
            n_trues = len([element for element in forest_implicant if element])
            if n_trees % 2 == 0:
                # Pair number of trees
                if prediction == 0:
                    limit = floor(n_trees / 2)
                else:
                    limit = floor((n_trees / 2)) + 1
            else:
                # Impair number of trees    
                limit = floor(n_trees / 2) + 1
                
            return n_trues >= limit

        reachable_classes = [tree.get_reachable_classes(implicant, prediction) for tree in self.forest]
        count_classes = [0] * self.n_classes
        for s in reachable_classes:
            for i in s:
                if i != prediction or len(s) == 1:
                    count_classes[i] += 1

        return all(count_classes[prediction] > count_classes[i] or i == prediction for i in range(self.n_classes))


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
            target_prediction = self._take_decisions_instance(instance)

        cnf = []
        # We add firsly the cardinality constraint dealing with the votes of the trees in the forest.
        # Each tree is represented by a new variable
        new_variables_atleast = [v for v in range(1 + n_original_variables, 1 + n_original_variables + self.n_trees)]

        if tree_encoding == Encoding.MUS:
            if self.n_trees % 2 == 0:
                # Pair number of trees
                if target_prediction == 0:
                    condition_atleast = floor(self.n_trees / 2) + 1
                else:
                    condition_atleast = floor(self.n_trees / 2)
            else:
                # Impair number of trees
                condition_atleast = floor(self.n_trees / 2) + 1     
        else:
            if self.n_trees % 2 == 0:
                # Pair number of trees
                if target_prediction == 0:
                    condition_atleast = floor(self.n_trees / 2)
                else:
                    condition_atleast = floor(self.n_trees / 2) + 1
            else:
                # Impair number of trees
                condition_atleast = floor(self.n_trees / 2) + 1     
        
        cardinality_encoding = EncType.seqcounter if cardinality_encoding == Encoding.SEQUENTIAL_COUNTER else EncType.totalizer
        atleast_clauses = CardEnc.atleast(lits=new_variables_atleast, encoding=cardinality_encoding, bound=condition_atleast,
                                          top_id=new_variables_atleast[-1]).clauses

        cnf.extend(atleast_clauses)

        # We secondly encode the trees
        for i, new_variable in enumerate(new_variables_atleast):
            current_tree = self.forest[i]

            if tree_encoding == Encoding.COMPLEMENTARY:
                clauses_for_l = current_tree.to_CNF(instance, target_prediction, format=False)
                clauses_for_not_l = current_tree.to_CNF(instance, target_prediction, format=False, inverse_coding=True)
                if current_tree._root.is_leaf():  #  special case when the tree is just a leaf value (clauses_for_l = [])
                    clauses_for_l.append([new_variable] if current_tree._root.is_prediction(target_prediction) else [-new_variable])
                else:
                    for clause in clauses_for_l:
                        clause.append(-new_variable)
                    for clause in clauses_for_not_l:
                        clause.append(new_variable)
                cnf.extend(clauses_for_l + clauses_for_not_l)
            elif tree_encoding == Encoding.SIMPLE:
                clauses_for_l = current_tree.to_CNF(instance, target_prediction, format=False)
                if current_tree._root.is_leaf():  # special case when the tree is just a leaf value (clauses_for_l = [])
                    clauses_for_l.append([new_variable] if current_tree._root.is_prediction(target_prediction) else [-new_variable])
                else:
                    for clause in clauses_for_l:
                        clause.append(-new_variable)
                cnf.extend(clauses_for_l)
            elif tree_encoding == Encoding.MUS:
                clauses_for_l = current_tree.to_CNF(instance, target_prediction, format=False, inverse_coding=True)
                if current_tree._root.is_leaf():  # special case when the tree is just a leaf value (clauses_for_l = [])
                    clauses_for_l.append([-new_variable] if current_tree._root.is_prediction(target_prediction) else [new_variable])
                else:
                    for clause in clauses_for_l:
                        clause.append(-new_variable)
                cnf.extend(clauses_for_l)
            else:
                assert False, "Bad parameter for " + str(tree_encoding) + " !"

        return CNFencoding.format(cnf)


    def to_CNF_sufficient_reason_multi_classes(self, instance, binary_representation, target_prediction):
        last_lit = len(binary_representation) + 1
        n_classes = self.n_classes
        n_trees = len(self.forest)
        hard_clauses = []

        # Init all additional literals except the one from cardinality constraints
        selectors = []
        for i in range(n_trees):
            selectors.append([last_lit + j for j in range(n_classes)])
            last_lit += n_classes

        unicity_challengers = [last_lit + j for j in range(n_classes)]
        last_lit += n_classes

        challengers = [last_lit + i for i in range(n_trees)]
        last_lit += n_trees

        # Encode each tree with the selector
        for i in range(n_trees):
            tree = self.forest[i]
            for k in range(n_classes):
                clauses = tree.to_CNF(instance, k, format=False)
                if tree._root.is_leaf():  # special case when the tree is just a leaf value (clauses_for_l = [])
                    clauses.append([selectors[i][k]] if tree._root.is_prediction(k) else [-selectors[i][k]])
                else:
                    for clause in clauses:
                        clause.append(-selectors[i][k])
                hard_clauses.extend(clauses)
                if k != target_prediction:
                    hard_clauses.append([-unicity_challengers[k], -selectors[i][k], challengers[i]])
                    hard_clauses.append([-unicity_challengers[k], selectors[i][k], -challengers[i]])
                else:
                    clauses = tree.to_CNF(instance, k, format=False, inverse_coding=True)
                    if tree._root.is_leaf():  # special case when the tree is just a leaf value (clauses_for_l = [])
                        clauses.append([-selectors[i][k]] if tree._root.is_prediction(k) else [selectors[i][k]])
                    else:
                        for clause in clauses:
                            clause.append(selectors[i][k])
                    hard_clauses.extend(clauses)
        # Step 2 : cardinality constraint unicity
        base_cls = []
        for k in range(n_classes):
            base_cls.append(unicity_challengers[k])
            for cl2 in range(k + 1, n_classes):
                hard_clauses.append([-unicity_challengers[k], -unicity_challengers[cl2]])
        hard_clauses.append(base_cls)
        hard_clauses.append([-unicity_challengers[target_prediction]])

        # step 3 : cardinality constraint target VS unicity
        lits = [-selectors[i][target_prediction] for i in range(n_trees)]
        lits.extend([challengers[i] for i in range(n_trees)])
        hard_clauses.extend(CardEnc.atleast(lits=lits, encoding=EncType.seqcounter, bound=n_trees,
                                            top_id=last_lit).clauses)
        return CNFencoding.format(hard_clauses)


    def to_CNF_majoritary_reason_multi_classes(self, instance, binary_representation, target_prediction):
        last_lit = len(binary_representation) + 1
        n_classes = self.n_classes
        n_trees = len(self.forest)
        hard_clauses = []
        # Init all additional literals except the one from cardinality constraints
        selectors = []
        for k in range(n_trees):
            selectors.append([last_lit + j for j in range(n_classes)])
            last_lit += n_classes

        for k in range(n_trees):
            tree = self.forest[k]
            for c in range(n_classes):
                if c == target_prediction:
                    clauses = tree.to_CNF(instance, c, format=False)
                    if tree._root.is_leaf():  # special case when the tree is just a leaf value (clauses_for_l = [])
                        clauses.append([selectors[k][c]] if tree._root.is_prediction(c) else [-selectors[k][c]])
                    else:
                        for clause in clauses:
                            clause.append(-selectors[k][c])
                else:
                    clauses = tree.to_CNF(instance, c, format=False, inverse_coding=True)
                    if tree._root.is_leaf():  # special case when the tree is just a leaf value (clauses_for_l = [])
                        clauses.append([-selectors[k][c]] if tree._root.is_prediction(c) else [selectors[k][c]])
                    else:
                        for clause in clauses:
                            clause.append(selectors[k][c])
                hard_clauses.extend(clauses)

        for k in range(n_classes):
            if k != target_prediction:
                lits = [-selectors[i][target_prediction] for i in range(n_trees)] + [selectors[i][k] for i in range(n_trees)]
                cnf = CardEnc.atmost(lits=lits, encoding=EncType.seqcounter, bound=n_trees - 1, top_id=last_lit).clauses
                last_lit = CNFencoding.compute_max_id_variable(cnf)
                hard_clauses.extend(cnf)

        return CNFencoding.format(hard_clauses)
