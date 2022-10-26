//
// Created by audemard on 22/04/2022.
//

#ifndef CPP_CODE_TREE_H
#define CPP_CODE_TREE_H

#include "Node.h"
#include <Python.h>

namespace PyLE {
    enum Kind_of_Tree_RF { WRONG, GOOD, CURRENTLY_WRONG};

    class Tree {
    public :

        unsigned int target_class;
        u_char *memory = nullptr;
        Node *root = nullptr;
        std::vector<Node *> all_nodes;
        Kind_of_Tree_RF status; // Useful only with RF : this tree hasn't the good class
        std::vector<bool>  used_to_explain; //  related to instance: true if the lit is used to explain the tree
        std::vector<int> used_lits;
        Tree(PyObject *tree_obj, Type _type){
          root = parse(tree_obj, _type);
        }

        void display(Type _type) { root->display(_type); std::cout << std::endl;}
        ~Tree();
        Node* parse(PyObject *tree_obj, Type _type);
        Node* parse_recurrence(PyObject *tree_obj, Type _type);
        int nb_nodes() { return root->nb_nodes();}

        // BT functions
        double compute_weight(std::vector<bool> &instance, std::vector<bool> &active_lits, bool get_min) {
            return root->compute_weight(instance, active_lits, get_min);
        }
        void initialize_BT(std::vector<bool> &instance, bool get_min);


        // RF functions
        bool is_implicant(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction) {
            used_lits.clear();

            return root->is_implicant(instance, active_lits, prediction, used_lits);
        }

        void initialize_RF(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction);

        void update_used_lits() {
            for(int i : used_lits)
                used_to_explain[i] = true;
        }
    };
}
#endif //CPP_CODE_TREE_H
