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
        Tree* tree;
        Tree* decision_rule;

        Rectifier(): tree(NULL), decision_rule(NULL){};

        void setTree(PyObject *tree_obj);
        void setDecisionRule(PyObject *tree_obj);
        
        void negatingDecisionRule();
    };
}


#endif //CPP_CODE_RECTIFIER_H
