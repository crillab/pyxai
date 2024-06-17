//
// Created by audemard on 22/04/2022.
//

#ifndef CPP_CODE_RECTIFIER_H
#define CPP_CODE_RECTIFIER_H

#include <Python.h>
#include <vector>
#include <map>
#include <algorithm>

#include "Tree.h"
#include "utils/TimerHelper.h"
#include "bcp/Propagator.h"
#include "bcp/Problem.h"
namespace pyxai {
    class Rectifier {
      public:
        std::vector<Tree*> trees;
        std::vector<Tree*> decision_rules;
        
        Propagator *theory_propagator = nullptr;

        Rectifier(): trees(), decision_rules(){};

        void addTree(PyObject *tree_obj);
        void addDecisionRule(PyObject *tree_obj);
        
        void negatingDecisionRules();
        void disjointTreesDecisionRule();
        void concatenateTreesDecisionRule();

        void simplifyTheory();
        void simplifyRedundant();

        int nNodes();

        void free();
    };
}


#endif //CPP_CODE_RECTIFIER_H
