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

void pyxai::Rectifier::setTree(PyObject *tree_obj) {
    tree = new Tree(tree_obj, pyxai::Classifier_RF);
}

void pyxai::Rectifier::setDecisionRule(PyObject *tree_obj) {
    decision_rule = new Tree(tree_obj, pyxai::Classifier_RF);
}

void pyxai::Rectifier::negatingDecisionRule() {
    decision_rule->display(pyxai::Classifier_RF);
    decision_rule->negating_tree();
    decision_rule->display(pyxai::Classifier_RF);
    
}





