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
#include <Python.h>


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
            leaf_value.weight = w;
            add_to_delete();
        }

        Node(int p, Tree *t) : lit(0), false_branch(nullptr), true_branch(nullptr), true_min(0), true_max(0), artificial_leaf(false), tree(t) {
            leaf_value.prediction = p;
            add_to_delete();
        }

        Node(int l, Node *f, Node *t) : lit(l), false_branch(f), true_branch(t), true_min(0), true_max(0), artificial_leaf(false), tree(f->tree) {
            add_to_delete();
        }

        Node(const Node* other){
                if (other != nullptr){
                    
                    lit = other->lit; 
                    leaf_value = other->leaf_value;
                    if (other->false_branch != nullptr)
                        false_branch = new Node(other->false_branch); 
                    else
                        false_branch = nullptr;
                    if (other->true_branch != nullptr)    
                        true_branch = new Node(other->true_branch);
                    else
                        true_branch = nullptr;
                    true_min = other->true_min;
                    true_max = other->true_max; 
                    artificial_leaf = other->artificial_leaf; 
                    tree = other->tree;
                    add_to_delete();
                }
        }

        void add_to_delete();

        inline void _delete(){
            if (is_leaf()){
                delete this;
            }else{
                true_branch->_delete();
                false_branch->_delete();
                delete this;
            }
        }

        inline int nNodes(Node* node){
            if (node->is_leaf()){
                return 1;
            }else{
                return 1 + nNodes(node->true_branch) + nNodes(node->false_branch);
            }
        }

        inline PyObject* toTuple(){
            
            if (is_leaf()){
                return PyLong_FromLong(leaf_value.prediction);
            }else{
                PyObject* tuple = PyTuple_New(2);
                PyObject *id = PyLong_FromLong(lit);
                PyObject* tuple_child = PyTuple_New(2);
                PyTuple_SET_ITEM(tuple_child, 0, false_branch->toTuple());
                PyTuple_SET_ITEM(tuple_child, 1, true_branch->toTuple());

                PyTuple_SET_ITEM(tuple, 0, id);
                PyTuple_SET_ITEM(tuple, 1, tuple_child);
                return tuple;
            }
            
        }


        inline void negating_tree(){
            if (is_leaf()) {
                if (leaf_value.prediction == 1){
                    leaf_value.prediction = 0;
                }else if (leaf_value.prediction == 0){
                    leaf_value.prediction = 1;
                }
            }else{
                false_branch->negating_tree();
                true_branch->negating_tree();
            }
        }

        inline void concatenateTreeDecisionRule(Node* decision_rule_root){
            if (true_branch->is_leaf()){
                if (true_branch->leaf_value.prediction == 1){
                    true_branch = new Node(decision_rule_root);
                    //true_branch = decision_rule_root;
                }
            }else{
                true_branch->concatenateTreeDecisionRule(decision_rule_root);
            }

            if (false_branch->is_leaf()){
                if (false_branch->leaf_value.prediction == 1){
                    false_branch = new Node(decision_rule_root);
                    //false_branch = decision_rule_root;
                }
            }else{
                false_branch->concatenateTreeDecisionRule(decision_rule_root);
            }
            
        }

        inline void disjointTreeDecisionRule(Node* decision_rule_root){
            if (true_branch->is_leaf()){
                if (true_branch->leaf_value.prediction == 0){
                    true_branch = new Node(decision_rule_root);
                    //true_branch = decision_rule_root;
                }
            }else{
                true_branch->disjointTreeDecisionRule(decision_rule_root);
            }

            if (false_branch->is_leaf()){
                if (false_branch->leaf_value.prediction == 0){
                    false_branch = new Node(decision_rule_root);
                    //false_branch = decision_rule_root;
                }
            }else{
                false_branch->disjointTreeDecisionRule(decision_rule_root);
            }
        }

        bool is_leaf() {
            return artificial_leaf || (false_branch == nullptr && true_branch == nullptr);
        }

        void display(Type _type) {
            if (is_leaf()) {
                std::cout << "[" << (_type == Classifier_BT ? leaf_value.weight : leaf_value.prediction) << "]";
            } else {
                std::cout << "[" << lit << ",";
                false_branch->display(_type);
                //std::cout << "\n";
                true_branch->display(_type);
                //std::cout << "\n";
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
