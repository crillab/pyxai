//
// Created by audemard on 22/04/2022.
//

#include "Tree.h"
#include<vector>

PyLE::Node *PyLE::Tree::parse(PyObject *tree_obj, Type _type) {
    //std::cout << "parse" << std::endl;

    Py_ssize_t size_obj = PyTuple_Size(tree_obj);
    if (size_obj != 2) {
        std::cout << "A"<<std::endl;
        PyErr_Format(PyExc_TypeError, "The size of the tuple have to be equal to 2 !");
        return NULL;
    }

    PyObject *target_class_obj = PyTuple_GetItem(tree_obj, 0);
    if (!PyLong_Check(target_class_obj)) {
        std::cout << "B"<<std::endl;
        PyErr_Format(PyExc_TypeError,
                     "The element of the tuple must be a integer representing the target class to evaluate !");
        return NULL;
    }

    target_class = PyLong_AsLong(target_class_obj);
    //std::cout << "target_class:" << target_class << std::endl;

    return parse_recurrence(PyTuple_GetItem(tree_obj, 1), _type);
}

PyLE::Node *PyLE::Tree::parse_recurrence(PyObject *tree_obj, Type _type) {
    Py_ssize_t size_obj = PyTuple_Size(tree_obj);

    if (size_obj != 3 && size_obj != 1) {
        std::cout << "C"<<std::endl;
        PyErr_Format(PyExc_TypeError, "The size of the tuple have to be equal to 3 if it is a complete tree or 1 if it is just one leaf value !");
        return NULL;
    }

    if (size_obj == 1){
      // it is a tree with only one leaf value !
      PyObject *value_obj = PyTuple_GetItem(tree_obj, 0);
      Node *tmp = _type == PyLE::BT ? new Node(PyFloat_AsDouble(value_obj)) : new Node((int)PyLong_AsLong(value_obj));
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
        left_node = _type == PyLE::BT ? new Node(PyFloat_AsDouble(left_obj)) : new Node((int)PyLong_AsLong(left_obj));
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
        right_node = _type == PyLE::BT ? new Node(PyFloat_AsDouble(right_obj)) : new Node((int)PyLong_AsLong(right_obj));
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


void PyLE::Tree::initialize_BT(std::vector<bool> &instance, bool get_min) {
    for(Node *n : all_nodes)
        n->artificial_leaf = false;
    root->reduce_with_instance(instance, get_min);

    // Do not traverse right branch // TODO CHeck
    //root->extremum_true_branch(true);
    //root->extremum_true_branch(false);
}


void PyLE::Tree::initialize_RF(std::vector<bool> &instance, std::vector<bool> &active_lits, int prediction) {
    status = PyLE::GOOD;
    if(used_to_explain.size() == 0)
        used_to_explain.resize( instance.size(), false);
    std::fill(used_to_explain.begin(), used_to_explain.end(), false);
    if(is_implicant(instance, active_lits, prediction) == false) // Do not try this tree : always wrong
        status = PyLE::WRONG;
    else
        update_used_lits();

}


