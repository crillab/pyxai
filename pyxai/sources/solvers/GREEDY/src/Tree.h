//
// Created by audemard on 22/04/2022.
//

#ifndef CPP_CODE_TREE_H
#define CPP_CODE_TREE_H

#include "Node.h"
#include "bcp/Propagator.h"
#include "constants.h"
#include <Python.h>

#if (__sun && __SVR4)
/* libnet should be using the standard type names, but in the short term
 * define our non-standard type names in terms of the standard names.
 */
#include <inttypes.h>
typedef uint8_t  u_int8_t;
typedef uint16_t u_int16_t;
typedef uint32_t u_int32_t;
typedef uint64_t u_int64_t;
typedef uint64_t u_int64_t;
#endif

namespace pyxai {
    enum Kind_of_Tree_RF { DEFINITIVELY_WRONG, GOOD, CURRENTLY_WRONG};
    class Node;

    class Tree {
    public :
        Type _type;
        unsigned int n_classes;
        unsigned int target_class;
        unsigned int *memory = nullptr;
        Node *root = nullptr;
        std::vector<Node *> all_nodes;
        Kind_of_Tree_RF status; // Useful only with Classifier_RF : this tree hasn't the good class
        std::vector<bool>  used_to_explain; //  related to instance: true if the lit is used to explain the tree
        std::vector<int> used_lits;
        Propagator *propagator = nullptr;
        std::set<Node*> to_delete;
        
        // Variables used to stored the comutation value during common is_impicant function
        // FOR Classifier_BT
        bool get_min;
        double current_weight;
        bool firstLeaf;
        double current_min_weight, current_max_weight; // For regression BT

        std::set<unsigned int> reachable_classes; // FOR Multiclasses Classifier_RF


        Tree(PyObject *tree_obj, Type _t): _type(_t) {
          root = parse(tree_obj, _t);
        }

        void display(Type _type);
        Node* parse(PyObject *tree_obj, Type _type);
        Node* parse_recurrence(PyObject *tree_obj, Type _type);
        int nb_nodes();

        PyObject* toTuple();

        void initialize_BT(std::vector<bool> &instance, bool get_min);


        bool is_implicant(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction);

        int nNodes();


        void initialize_RF(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction);

        bool equalTree(Node* node1, Node* node2);

        void negating_tree();
        void concatenateTreeDecisionRule(Tree* decision_rule);
        void disjointTreeDecisionRule(Tree* decision_rule);

        void improvedRectification(std::vector<int>* conditions, int& label);
        void _improvedRectification(Node* node, Node* parent, int come_from, std::vector<int>* stack, std::vector<int>* appeared_conditions, std::vector<int>* in_conditions, std::vector<int>* conditions, int& label);
        
        void simplifyTheory();
        void free();
        void simplifyRedundant();
        bool _simplifyRedundant(Node* root, Node* node, std::vector<int>* path, int come_from, Node* previous_node, Node* previous_previous_node);

        Node* _simplifyTheory(Node* node, std::vector<Lit>* stack, Node* parent, int come_from, Node* root);
        std::vector<bool>* isNodeConsistent(Node* node, std::vector<Lit>* stack);

        void update_used_lits() {
            for(int i : used_lits)
                used_to_explain[i] = true;
        }
    };
}
#endif //CPP_CODE_TREE_H
