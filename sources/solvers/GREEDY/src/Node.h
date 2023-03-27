//
// Created by audemard on 22/04/2022.
//

#ifndef CPP_CODE_NODE_H
#define CPP_CODE_NODE_H


#include <iostream>
#include <vector>
#include <set>
#include "bcp/Propagator.h"
#include "Tree.h"
#include "constants.h"



namespace pyxai {

    class Tree;
    class Node {
    public:
        int lit;
        union {
            double weight;
            int prediction;
        } leaf_value;

        Node *false_branch, *true_branch;
        double true_min, true_max; // The min and max possible values for true branch
        bool artificial_leaf = false;
        Tree *tree;

        Node(double w, Tree *t): lit(0), false_branch(nullptr), true_branch(nullptr), true_min(0), true_max(0), artificial_leaf(false), tree(t) {
            leaf_value = {.weight=w};
        }

        Node(int p, Tree *t) : lit(0), false_branch(nullptr), true_branch(nullptr), true_min(0), true_max(0), artificial_leaf(false), tree(t) {
            leaf_value = {.prediction=p};
        }

        Node(int l, Node *f, Node *t) : lit(l), false_branch(f), true_branch(t), true_min(0), true_max(0), artificial_leaf(false), tree(f->tree) {}


        bool is_leaf() {
            return artificial_leaf || (false_branch == nullptr && true_branch == nullptr);
        }

        void display(Type _type) {
            if (is_leaf()) {
                std::cout << "[" << (_type == Classifier_BT ? leaf_value.weight : leaf_value.prediction) << "]";
            } else {
                std::cout << "[" << lit << ",";
                false_branch->display(_type);
                std::cout << "\n";
                true_branch->display(_type);
                std::cout << "\n";
                std::cout << "]";
            }
        }

        int nb_nodes();

        // Methods only related to Classifier_BT
        double compute_weight(std::vector<bool> &instance, std::vector<bool> &active_lits, bool get_min);

        void reduce_with_instance(std::vector<bool> &instance, bool get_min);

        double extremum_true_branch(bool get_min); // The extremum value (min or max) of the true branch

        // Methods onlu related to Classifier_RF
        void is_implicant(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction);

        void is_implicant_multiclasses(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction,
                                       std::set<unsigned int> &reachable_classes,
                                       Propagator *theory_propagator);

        void performOnLeaf();

    };
}

#endif //CPP_CODE_NODE_H
