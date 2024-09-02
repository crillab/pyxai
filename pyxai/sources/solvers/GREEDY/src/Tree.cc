//
// Created by audemard on 22/04/2022.
//

#include "Tree.h"
#include <algorithm>
#include <vector>
#include <stdexcept>

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

void pyxai::Tree::free(){
    for (Node* node:to_delete){
        delete node;
    }
}


const int NOT_APPEARED = -1;

const int NOT_IN_CONDITION = -1;
const int IN_CONDITION_NEGATIVELY = 0;
const int IN_CONDITION_POSITIVELY = 1;

void pyxai::Tree::improvedRectification(std::vector<int>* conditions, int& label){
    
    int max = 0;
    for (Node* node: all_nodes){
        if (!node->is_leaf()){
            if (abs(node->lit) > max) max = abs(node->lit);
        }
    }
    
    for (int& condition: *conditions){if (abs(condition) > max) max = abs(condition);}
    
    std::vector<int>* in_conditions = new std::vector<int>(max+1, NOT_IN_CONDITION);
    
    for (int& condition: *conditions){
        if (condition > 0) (*in_conditions)[abs(condition)] = IN_CONDITION_POSITIVELY;
        else (*in_conditions)[abs(condition)] = IN_CONDITION_NEGATIVELY;
    }
    
    std::vector<int>* appeared_conditions = new std::vector<int>(max+1, NOT_APPEARED);
    std::vector<int>* stack = new std::vector<int>();
    
    _improvedRectification(root, nullptr, -1, stack, appeared_conditions, in_conditions, conditions, label);
    
    delete in_conditions;
    delete appeared_conditions;
    delete stack;
    
}




/* in_conditionss: variable -> in the condition
 * Value: 
 * -1: NOT_IN_CONDITION: not in the condition
 *  0: IN_CONDITION_NEGATIVELY: in the condition negatively
 *  1: IN_CONDITION_POSITIVELY: in the condition positively
 */
void pyxai::Tree::_improvedRectification(Node* node, Node* parent, int come_from, std::vector<int>* stack, std::vector<int>* appeared_conditions, std::vector<int>* in_conditions, std::vector<int>* conditions, int& label){
    
    if (node->is_leaf()){
        int label_leaf = node->leaf_value.prediction; 
        /*std::cout << std::endl;
        std::cout << "**********************************" << std::endl;
        std::cout << "label_leaf:" << label_leaf << std::endl;
        std::cout << "come_from:" << come_from << std::endl;
        
        std::cout << "stack literals:" << std::endl;
        for (int value: *stack){
            std::cout << value << " ";
        }*/
        if (label != label_leaf){ // take into account the multi-classes
            // We have to do a change here: add the missing nodes of the rectification for this leaf 
            // Warning: if no missing nodes, we have may be change the classification of the leave. 
            
            // 1: Compute the missing nodes
            std::vector<int> missing_nodes;
            for (int& literal: *conditions){
                if ((*appeared_conditions)[abs(literal)] == NOT_APPEARED)missing_nodes.push_back(literal);
            }
            if (missing_nodes.size() > 0){
                // We have some missing node: we construct the sub-tree for the rectification with these nodes
                //std::cout << std::endl;
                //std::cout << "missing_nodes len:" << missing_nodes.size() << std::endl;
                
                //std::cout << "missing nodes:" << std::endl;
                //for (int& value: missing_nodes){
                //    std::cout << value << " ";
                //}
                //std::cout << std::endl;
                //std::cout << "missing_nodes len:" << missing_nodes.size() << std::endl;
                
                // Start with the last node: 
                int last_missing_node = missing_nodes[missing_nodes.size()-1];
                Node* right_leaf = new Node(last_missing_node > 0 ? label : label_leaf, node->tree);
                Node* left_leaf = new Node(last_missing_node <= 0 ? label : label_leaf, node->tree);
                Node* node = new Node(abs(last_missing_node), left_leaf, right_leaf);
                
                // Now we add the following
                for (int i = missing_nodes.size()-2; i >= 0; i--){
                    Node* leaf = new Node(label_leaf, node->tree); 
                    if (missing_nodes[i] > 0){
                        node = new Node(abs(missing_nodes[i]), leaf, node);
                    }else{
                        node = new Node(abs(missing_nodes[i]), node, leaf);
                    }
                }
                //node->display(Classifier_RF);
                //std::cout << std::endl;

                // Now we put this subtree in the tree
                if (come_from == 0){// come from the left side
                    parent->false_branch = node;
                }else if (come_from == 1){// come from the right side
                    parent->true_branch = node;
                }else{
                    std::cout << "Not implemented error: come from root" << std::endl;
                    exit(0);
                }
            }else{
                // We have no missing node: we have to change the label
                //std::cout << "No missing node" << std::endl;
                node->leaf_value.prediction = label;
                //exit(0);
            }
            //for (int& value: missing_nodes){
            //    
            //}
        }
        return;
    }
    //std::cout << "Start rectification:" << node->lit << std::endl;
    // appeared_conditions for a variable is the position in the stack 
    
    if ((*in_conditions)[abs(node->lit)] == NOT_IN_CONDITION){
        //std::cout << "NOT_IN_CONDITION" << std::endl;
        stack->push_back(-abs(node->lit));
        _improvedRectification(node->false_branch, node, 0, stack, appeared_conditions, in_conditions, conditions, label);
        stack->pop_back();

        stack->push_back(abs(node->lit));
        _improvedRectification(node->true_branch, node, 1, stack, appeared_conditions, in_conditions, conditions, label);
        stack->pop_back();

    }else if ((*in_conditions)[abs(node->lit)] == IN_CONDITION_NEGATIVELY){
        //std::cout << "IN_CONDITION_NEGATIVELY" << std::endl;
        
        (*appeared_conditions)[abs(node->lit)] = stack->size();
        stack->push_back(-abs(node->lit));
        _improvedRectification(node->false_branch, node, 0, stack, appeared_conditions, in_conditions, conditions, label);
        stack->pop_back();
        
        (*appeared_conditions)[abs(node->lit)] = NOT_APPEARED;
    }else if ((*in_conditions)[abs(node->lit)] == IN_CONDITION_POSITIVELY){
        //std::cout << "IN_CONDITION_POSITIVELY" << std::endl;
        (*appeared_conditions)[abs(node->lit)] = stack->size();
        
        stack->push_back(abs(node->lit));
        _improvedRectification(node->true_branch, node, 1, stack, appeared_conditions, in_conditions, conditions, label);
        stack->pop_back();
        
        (*appeared_conditions)[abs(node->lit)] = NOT_APPEARED;    
    }

    
    
    
    
}

void pyxai::Tree::simplifyTheory(){
    std::vector<Lit>* vec = new std::vector<Lit>();
    root = _simplifyTheory(root, vec, nullptr, -1, root);
    delete vec;
}

void pyxai::Tree::simplifyRedundant(){
    std::vector<int>* path = new std::vector<int>();
    while (_simplifyRedundant(root, root, path, -1, nullptr, nullptr) == true);
    delete path;
    if (equalTree(root->false_branch, root->true_branch) == true){
        root = root->false_branch;
    }
}

bool pyxai::Tree::equalTree(Node* node1, Node* node2){
    if (node1->is_leaf() && node2->is_leaf()){
        return node1->leaf_value.prediction == node2->leaf_value.prediction;
    }
    if ((node1->is_leaf() && !node2->is_leaf())||(!node1->is_leaf() && node2->is_leaf())){
        return false;
    }

    if (node1->lit != node2->lit){
        return false;
    }
    return equalTree(node1->false_branch, node2->false_branch) && equalTree(node1->true_branch, node2->true_branch);
}

bool pyxai::Tree::_simplifyRedundant(Node* root, Node* node, std::vector<int>* path, int come_from, Node* previous_node, Node* previous_previous_node){
    bool res_1 = false;
    bool res_2 = false;
    bool change = false;

    if (previous_node != nullptr){
        int literal = (come_from == 1) ? node->lit: -node->lit;
        
        if (std::find(path->begin(), path->end(), literal) != path->end() ){
            if (path->back() < 0){
                if (previous_previous_node != nullptr){
                    previous_previous_node->false_branch = node;
                    change = true;
                }
            }else if (path->back() > 0){
                if (previous_previous_node != nullptr){
                    previous_previous_node->true_branch = node;
                    change = true;
                }
            }
        }
        path->push_back(literal);
    }

    if (!node->is_leaf()){
        if (equalTree(node->false_branch, node->true_branch) == true){
            if (come_from == 0){
                if (previous_node != nullptr){
                    previous_node->false_branch = node->false_branch;
                    change = true;
                }
            }else if (come_from == 1){
                if (previous_node != nullptr){
                    previous_node->true_branch = node->true_branch;
                    change = true;
                }
            }   
        }
        std::vector<int>* copy_path_1 = new std::vector<int>(*path);  
        std::vector<int>* copy_path_2 = new std::vector<int>(*path);  
        
        res_1 = _simplifyRedundant(root, node->false_branch, copy_path_1, 0, node, previous_node);
        res_2 = _simplifyRedundant(root, node->true_branch, copy_path_2, 1, node, previous_node);
        delete copy_path_1;
        delete copy_path_2;
    }
    return res_1 || res_2 || change;    
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
    // Check consistency on the left
    stack->push_back(~lit);
    bool ret_left = propagator->propagate_assumptions(*stack);
        
    /*if (ret_left == false){
        std::cout << "left_consistent:";
        for (unsigned int i = 0; i < stack->size();i++){
            std::cout << (*stack)[i] << " ";
        } 
        std::cout << std::endl;
    }*/
    stack->pop_back();
    results->push_back(ret_left);

    // Check consistency on the right
    stack->push_back(lit);
    bool ret_right = propagator->propagate_assumptions(*stack);
    /*std::cout << "ret_right:" << ret_right << std::endl;
    if (ret_right == false){
        std::cout << "right_consistent:";
        for (unsigned int i = 0; i < stack->size();i++){
            std::cout << (*stack)[i] << " ";
        } 
        std::cout << std::endl;
    }*/
    
    stack->pop_back();
    results->push_back(ret_right);
    return results;
}          

/* come_from: -1 => None or 0 or 1 
*
*/
pyxai::Node* pyxai::Tree::_simplifyTheory(Node* node, std::vector<Lit>* stack, Node* parent, int come_from, Node* root){
    if (node->is_leaf()){
        return root;
    }
    
    Lit lit_positif = Lit::makeLitTrue(abs(node->lit));
    Lit lit_negatif = Lit::makeLitFalse(abs(node->lit));
    std::vector<bool>* results = isNodeConsistent(node, stack);
    bool left_consistent = (*results)[0];
    bool right_consistent = (*results)[1];
    //delete results;

    if ((left_consistent == false) && (right_consistent == false)){
        //Impossible Case
        throw std::invalid_argument("Impossible Case : both are inconsistent");
    }else if ((left_consistent == true) && (right_consistent == true)){
        stack->push_back(lit_negatif);
        root = _simplifyTheory(node->false_branch, stack, node, 0, root);
        stack->pop_back();
        stack->push_back(lit_positif);
        root = _simplifyTheory(node->true_branch, stack, node, 1, root);
        stack->pop_back();
        return root;
    }else if (left_consistent == false){
        if (come_from == -1){
            // The root change
            //stack->push_back(lit);
            return _simplifyTheory(node->true_branch, stack, nullptr, -1, node->true_branch);
            //stack->pop_back();
        }else if(come_from == 0){
            // Replace the node
            /*if (parent->false_branch != node->true_branch){
                to_delete.push_back(parent->false_branch);
            }*/
            //std::cout << "left inconsistent come from 0" << std::endl;
            parent->false_branch = node->true_branch;
            //stack->push_back(lit);
            return _simplifyTheory(parent->false_branch, stack, parent, 0, root);
            //stack->pop_back();
        }else if(come_from == 1){
            // Replace the node
            /*if (parent->true_branch != node->true_branch){
                to_delete.push_back(parent->true_branch);
            }*/
            //std::cout << "left inconsistent come from 1" << std::endl;
            parent->true_branch = node->true_branch;
            //stack->push_back(lit);
            return _simplifyTheory(parent->true_branch, stack, parent, 1, root);
            //stack->pop_back();
        }
    }else if (right_consistent == false){
        if (come_from == -1){
            // The root change
            // stack->push_back(~lit);
            return _simplifyTheory(node->false_branch, stack, nullptr, -1, node->false_branch);
            // stack->pop_back();
        }else if(come_from == 0){
            // Replace the node
            /*if (parent->false_branch != node->false_branch){
                to_delete.push_back(parent->false_branch);
            }*/
            //std::cout << "right inconsistent come from 0" << std::endl;
            parent->false_branch = node->false_branch;
            //stack->push_back(~lit);
            return _simplifyTheory(parent->false_branch, stack, parent, 0, root);
            //stack->pop_back();
        }else if(come_from == 1){
            // Replace the node
            /*if (parent->true_branch != node->false_branch){
                to_delete.push_back(parent->true_branch);
            }*/
            //std::cout << "right inconsistent come from 1" << std::endl;
            parent->true_branch = node->false_branch;
            //stack->push_back(~lit);
            //return root;
            return _simplifyTheory(parent->true_branch, stack, parent, 1, root);
            //stack->pop_back();
        }
    }else{
        throw std::invalid_argument("Impossible Case");
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
        //std::cout << "C"<<std::endl;
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
        //std::cout << p << std::endl;

        std::cout << "Error:" << PyLong_AsLong(left_obj) << std::endl;
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
        //std::cout << p << std::endl;
        std::cout << "Error:" << PyLong_AsLong(right_obj) << std::endl;
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
