//
// Created by audemard on 22/04/2022.
//

#include "Rectifier.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <random>
#include <iostream>
#include <chrono>
#include "Tree.h"
#include "bcp/ProblemTypes.h"

void pyxai::Rectifier::addTree(PyObject *tree_obj) {
    trees.clear();
    Tree *tree = new Tree(tree_obj, pyxai::Classifier_RF);
    trees.push_back(tree);
}

void pyxai::Rectifier::setDecisionRule(PyObject *tree_obj) {
    free(decision_rule);
    decision_rule = new Tree(tree_obj, pyxai::Classifier_RF);
}

int pyxai::Rectifier::nNodes() {
    int sum = 0;
    for (Tree *tree: trees) {sum = sum + tree->nNodes();}
    return sum;
}

void pyxai::Rectifier::negatingDecisionRule() {
    decision_rule->display(pyxai::Classifier_RF);
    decision_rule->negating_tree();
    decision_rule->display(pyxai::Classifier_RF);
    
}

void pyxai::Rectifier::disjointTreesDecisionRule() {
    decision_rule->display(pyxai::Classifier_RF);
    
    for (Tree *tree: trees) {
        tree->display(pyxai::Classifier_RF);
        tree->disjointTreeDecisionRule(decision_rule);
        tree->display(pyxai::Classifier_RF);
    }
    
}


void pyxai::Rectifier::concatenateTreesDecisionRule() {
    decision_rule->display(pyxai::Classifier_RF);
    
    for (Tree *tree: trees) {
        tree->display(pyxai::Classifier_RF);
        tree->concatenateTreeDecisionRule(decision_rule);
        tree->display(pyxai::Classifier_RF);
    }
}

void pyxai::Rectifier::simplifyTheory() {
    for (Tree *tree: trees) tree->simplifyTheory();
}








