//
// Created by audemard on 22/04/2022.
//

#ifndef CPP_CODE_EXPLAINER_H
#define CPP_CODE_EXPLAINER_H

#include <Python.h>
#include <vector>
#include<algorithm>
#include "Tree.h"
#include "utils/TimerHelper.h"

namespace PyLE {
    class Explainer {
      public:
        int n_classes;
        Type _type;
        int n_iterations;
        int time_limit; //in seconds
        std::vector<double> weights; //useful only for multiclasses (BT)
        int try_to_remove;
        std::vector<double> excluded_features;

        Explainer(int _n_classes, Type t) : n_classes(_n_classes), _type(t), n_iterations(50), time_limit(0) {
            if(n_classes > 2)
                for(int i = 0; i < n_classes; i++) weights.push_back(0.0);
        }


        void addTree(PyObject *tree_obj);
        std::vector<Tree*> trees;
        void compute_reason_conditions(std::vector<int> &instance, int prediction, std::vector<int> &reason, long seed);
        void compute_reason_features(std::vector<int> &instance, std::vector<int> &features, int prediction, std::vector<int> &reason);
        
        bool is_implicant(std::vector<bool> &instance, std::vector<bool> &active_lits, unsigned int prediction);

        bool is_implicant_BT(std::vector<bool> &instance, std::vector<bool> &active_lits, unsigned int prediction);
        bool is_implicant_RF(std::vector<bool> &instance, std::vector<bool> &active_lits, unsigned int prediction);

        inline void set_n_iterations(int _n_iterations){n_iterations = _n_iterations;}
        inline void set_time_limit(int _time_limit){time_limit = _time_limit;}
        inline bool is_specific(int l) {return std::find(excluded_features.begin(), excluded_features.end(), l) == excluded_features.end();}
    };
}

//tree_specific:  (-1, -2, -3, -4, 5, -6, -7, -9, -10, -11, -14, -16, -18, -19, -21, -23, -24, -25, -27, -29, -32, -33, -34, -37, -40)

#endif //CPP_CODE_EXPLAINERBT_H
