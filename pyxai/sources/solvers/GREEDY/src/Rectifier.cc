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
    delete decision_rule;
    decision_rule = new Tree(tree_obj, pyxai::Classifier_RF);
}

int pyxai::Rectifier::nNodes() {
    int sum = 0;
    for (Tree *tree: trees) {sum = sum + tree->nNodes();}
    return sum;
}

void pyxai::Rectifier::negatingDecisionRule() {
    decision_rule->negating_tree();
}

void pyxai::Rectifier::free(){
    for (Tree *tree: trees) {
        tree->free();
        delete tree;
    }
    trees.clear();
}

void pyxai::Rectifier::simplifyRedundant(){
    for (Tree *tree: trees) {tree->simplifyRedundant();}
}

void pyxai::Rectifier::disjointTreesDecisionRule() {
    for (Tree *tree: trees) {tree->disjointTreeDecisionRule(decision_rule);}
}


void pyxai::Rectifier::concatenateTreesDecisionRule() {
    for (Tree *tree: trees) {tree->concatenateTreeDecisionRule(decision_rule);}
}

void pyxai::Rectifier::simplifyTheory() {
    for (Tree *tree: trees) tree->simplifyTheory();
}








