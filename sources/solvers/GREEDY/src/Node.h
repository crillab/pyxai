//
// Created by audemard on 22/04/2022.
//

#ifndef CPP_CODE_NODE_H
#define CPP_CODE_NODE_H

#include <iostream>
#include<vector>


namespace PyLE {
    enum Type {BT, RF};
    class Node {
    public:
        int lit;
        union {
            double weight;
            int prediction;
        } leaf_value;

        Node *false_branch, *true_branch;
        double true_min, true_max; // The min and max possible values for true branch
        bool artificial_leaf;

        Node(double w) : lit(0), false_branch(nullptr), true_branch(nullptr), artificial_leaf(false) {leaf_value.weight =w;}
        Node(int p) : lit(0), false_branch(nullptr), true_branch(nullptr), artificial_leaf(false) {leaf_value.prediction = p;}
        Node(int l, Node *f, Node *t) : lit(l), false_branch(f), true_branch(t), artificial_leaf(false) {}


        bool is_leaf() { return artificial_leaf || (false_branch == nullptr && true_branch == nullptr);}
        void display(Type _type) {
            if (is_leaf()){
              std::cout << "[" << (_type == BT ? leaf_value.weight : leaf_value.prediction) << "]";
            }else{
              std::cout << "[" << lit << ",";
              false_branch->display(_type); std::cout << "\n";
              true_branch->display(_type); std::cout << "\n";
              std::cout << "]";
            }
        }

        int nb_nodes();

        // Methods only related to BT
        double compute_weight(std::vector<bool> &instance, std::vector<bool> &active_lits, bool get_min);
        void reduce_with_instance(std::vector<bool> &instance, bool get_min);
        double extremum_true_branch(bool get_min); // The extremum value (min or max) of the true branch

        // Methods onlu related to RF
        bool is_implicant(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction, std::vector<int> &used_lits);
    };
}

#endif //CPP_CODE_NODE_H
