//
// Created by audemard on 22/04/2022.
//

#include "Tree.h"
#include<vector>

void pyxai::Tree::negating_tree() {
    root->negating_tree();
}

void pyxai::Tree::concatenateTreeDecisionRule(Tree* decision_rule){
    root->concatenateTreeDecisionRule(decision_rule->root);
}
        
void pyxai::Tree::disjointTreeDecisionRule(Tree* decision_rule){
    root->disjointTreeDecisionRule(decision_rule->root);
}

int pyxai::Tree::nNodes(){
    return root->nNodes(root);
}

void pyxai::Tree::simplifyTheory(){
    std::cout << "simplifyTheory" << std::endl;
    std::vector<Lit>* vec = new std::vector<Lit>();
    root = _simplifyTheory(root, vec, nullptr, -1, root);

}

std::vector<bool>* pyxai::Tree::isNodeConsistent(Node* node, std::vector<Lit>* stack){
    if (node->is_leaf()){
        std::vector<bool>* results = new std::vector<bool>();
        results->push_back(false);
        results->push_back(false);
        return results;
    }
    std::vector<bool>* results = new std::vector<bool>();
    Lit lit = node->lit > 0 ? Lit::makeLitTrue(node->lit) : Lit::makeLitFalse(-node->lit);
    std::cout << "left:" << std::endl;
    // Check consistency on the left
    stack->push_back(~lit);
    bool ret_left = propagator->propagate_assumptions(*stack);
    stack->pop_back();
    results->push_back(ret_left);

    // Check consistency on the right
    std::cout << "right:" << std::endl;
    stack->push_back(lit);
    bool ret_right = propagator->propagate_assumptions(*stack);
    stack->pop_back();
    results->push_back(ret_right);
    return results;
}          

/* come_from: -1 => None or 0 or 1 
*
*/
pyxai::Node* pyxai::Tree::_simplifyTheory(Node* node, std::vector<Lit>* stack, Node* parent, int come_from, Node* root){
    std::cout << "simplifyTheory2" << std::endl;
    if (node->is_leaf())return root;
    Lit lit = node->lit > 0 ? Lit::makeLitTrue(node->lit) : Lit::makeLitFalse(-node->lit);
    std::vector<bool>* results = isNodeConsistent(node, stack);
    bool left_consistent = (*results)[0];
    bool right_consistent = (*results)[1];

    if (left_consistent == true){
        // The left part is consistent, simplify recursively
        stack->push_back(~lit);
        root = _simplifyTheory(node->false_branch, stack, node, 0, root);
        stack->pop_back();
    }else{
        // The left part is inconsistent, replace this node with the right
        if (come_from == -1){
            // The root change
            stack->push_back(lit);
            root = _simplifyTheory(node->true_branch, stack, nullptr, -1, node->true_branch);
            stack->pop_back();
        }else if(come_from == 0){
            // Replace the node
            parent->false_branch = node->true_branch;
            stack->push_back(lit);
            root = _simplifyTheory(node->true_branch, stack, parent, 0, root);
            stack->pop_back();
        }else if(come_from == 1){
            // Replace the node
            parent->true_branch = node->true_branch;
            stack->push_back(lit);
            root = _simplifyTheory(node->true_branch, stack, parent, 1, root);
            stack->pop_back();
        }
    }

    if (right_consistent == true){
        // The left part is consistent, simplify recursively
        stack->push_back(lit);
        root = _simplifyTheory(node->true_branch, stack, node, 1, root);
        stack->pop_back();
    }else{
        // The left part is inconsistent, replace this node with the right
        if (come_from == -1){
            // The root change
            stack->push_back(~lit);
            root = _simplifyTheory(node->false_branch, stack, nullptr, -1, node->false_branch);
            stack->pop_back();
        }else if(come_from == 0){
            // Replace the node
            parent->false_branch = node->false_branch;
            stack->push_back(~lit);
            root = _simplifyTheory(node->false_branch, stack, parent, 0, root);
            stack->pop_back();
        }else if(come_from == 1){
            // Replace the node
            parent->true_branch = node->false_branch;
            stack->push_back(~lit);
            root = _simplifyTheory(node->false_branch, stack, parent, 1, root);
            stack->pop_back();
        }
    }
    return root;
}

PyObject* pyxai::Tree::toTuple(){
    return root->toTuple();
}

pyxai::Node *pyxai::Tree::parse(PyObject *tree_obj, Type _type) {
    //std::cout << "parse" << std::endl;

    Py_ssize_t size_obj = PyTuple_Size(tree_obj);
    if (size_obj != 2) {
        PyErr_Format(PyExc_TypeError, "The size of the tuple have to be equal to 2 !");
        return NULL;
    }

    PyObject *target_class_obj = PyTuple_GetItem(tree_obj, 0);
    if (!PyLong_Check(target_class_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The element of the tuple must be a integer representing the target class to evaluate !");
        return NULL;
    }

    target_class = PyLong_AsLong(target_class_obj);
    //std::cout << "target_class:" << target_class << std::endl;

    return parse_recurrence(PyTuple_GetItem(tree_obj, 1), _type);
}

pyxai::Node *pyxai::Tree::parse_recurrence(PyObject *tree_obj, Type _type) {
    Py_ssize_t size_obj = PyTuple_Size(tree_obj);

    if (size_obj != 3 && size_obj != 1) {
        std::cout << "C"<<std::endl;
        PyErr_Format(PyExc_TypeError, "The size of the tuple have to be equal to 3 if it is a complete tree or 1 if it is just one leaf value !");
        return NULL;
    }

    if (size_obj == 1){
      // it is a tree with only one leaf value !
      PyObject *value_obj = PyTuple_GetItem(tree_obj, 0);
      Node *tmp = _type == pyxai::Classifier_BT || _type == Regression_BT ? new Node(PyFloat_AsDouble(value_obj), this) : new Node((int)PyLong_AsLong(value_obj), this);
      all_nodes.push_back(tmp);
      return tmp;
    }

    PyObject *value_obj = PyTuple_GetItem(tree_obj, 0);
    PyObject *left_obj = PyTuple_GetItem(tree_obj, 1);
    PyObject *right_obj = PyTuple_GetItem(tree_obj, 2);

    int value = PyLong_AsLong(value_obj);
    //std::cout << "value:" << value << std::endl;
    Node *left_node;
    Node *right_node;

    if (PyTuple_Check(left_obj)) {
        left_node = parse_recurrence(left_obj, _type);
    } else if (PyFloat_Check(left_obj) || PyLong_Check(left_obj)) {
        left_node = _type == Classifier_BT || _type == Regression_BT ? new Node(PyFloat_AsDouble(left_obj), this) : new Node((int)PyLong_AsLong(left_obj), this);
        all_nodes.push_back(left_node);
    } else {
        const char* p = Py_TYPE(left_obj)->tp_name;
        std::cout << p << std::endl;

        std::cout << "err:" << PyLong_AsLong(left_obj) << std::endl;
        PyErr_Format(PyExc_TypeError, "Error during passing: this element have to be float/int or tuple !");
        return NULL;
    }

    if (PyTuple_Check(right_obj)) {
        right_node = parse_recurrence(right_obj, _type);
    } else if (PyFloat_Check(right_obj) || PyLong_Check(right_obj)) {
        right_node = _type == pyxai::Classifier_BT || _type == Regression_BT ? new Node(PyFloat_AsDouble(right_obj), this) : new Node((int)PyLong_AsLong(right_obj), this);
        all_nodes.push_back(right_node);
    } else {
        const char* p = Py_TYPE(right_obj)->tp_name;
        std::cout << p << std::endl;
        std::cout << "err:" << PyLong_AsLong(right_obj) << std::endl;
        PyErr_Format(PyExc_TypeError, "Error during passing: this element have to be float/int or tuple !");
        return NULL;
    }
    Node *tmp = new Node(value, left_node, right_node);
    all_nodes.push_back(tmp);
    return tmp;
}



void pyxai::Tree::initialize_BT(std::vector<bool> &instance, bool get_min) {
    for(Node *n : all_nodes)
        n->artificial_leaf = false;
    root->reduce_with_instance(instance, get_min);

    // Do not traverse right branch // TODO CHeck
    //root->extremum_true_branch(true);
    //root->extremum_true_branch(false);
}


void pyxai::Tree::initialize_RF(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction) {
    status = pyxai::GOOD;
    if(used_to_explain.empty())
        used_to_explain.resize( instance.size(), false);
    std::fill(used_to_explain.begin(), used_to_explain.end(), false);
    if(is_implicant(instance, active_lits, prediction)) // Do not try this tree : always wrong
        update_used_lits();
    else
        status = pyxai::DEFINITIVELY_WRONG;
}

bool pyxai::Tree::is_implicant(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction) {
    used_lits.clear();
    root->is_implicant(instance, active_lits, prediction);
    return true;
}

void pyxai::Tree::display(Type _type) { root->display(_type); std::cout << std::endl;}

int pyxai::Tree::nb_nodes() { return root->nb_nodes();}
