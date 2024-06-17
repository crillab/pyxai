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
    Tree *tree = new Tree(tree_obj, pyxai::Classifier_RF);
    trees.push_back(tree);
}

void pyxai::Rectifier::addDecisionRule(PyObject *tree_obj) {
    Tree *tree = new Tree(tree_obj, pyxai::Classifier_RF);
    decision_rules.push_back(tree);
}

int pyxai::Rectifier::nNodes() {
    int sum = 0;
    for (Tree *tree: trees) {sum = sum + tree->nNodes();}
    return sum;
}

void pyxai::Rectifier::negatingDecisionRules() {
    for (Tree *decision_rule: decision_rules) {decision_rule->negating_tree();}
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
    for (unsigned int i = 0; i < trees.size(); i++){
        trees[i]->disjointTreeDecisionRule(decision_rules[i]);
    }
}

void pyxai::Rectifier::concatenateTreesDecisionRule() {
    for (unsigned int i = 0; i < trees.size(); i++){
        trees[i]->concatenateTreeDecisionRule(decision_rules[i]);
    }
}

void pyxai::Rectifier::simplifyTheory() {
    for (Tree *tree: trees) tree->simplifyTheory();
}








