//
// Created by audemard on 22/04/2022.
//

#ifndef CPP_CODE_WRAPPER_H
#define CPP_CODE_WRAPPER_H

#include <Python.h>
#include "pythonToCMemory.h"

#include "Node.h"
#include "TreeBasedExplainer.h"

#include "Rectifier.h"

#include <sstream>
#include <limits>
#include <iostream>
#include <string>
#include <vector>

const double PRECISION = 0.0000001;

using namespace pyxai;




// PyObject *new_neural_network(PyObject *self, PyObject *args) {
//     PyObject *classes_obj;
//     char *activation_obj;
//     char *out_activation_obj;
//     PyObject *coefficients_obj;
//     PyObject *intercept_obj;
//     std::cout << "neural_network c++" << std::endl;
//     if (!PyArg_ParseTuple(args, "OssOO", &classes_obj, &activation_obj, &out_activation_obj, &coefficients_obj, &intercept_obj))
//         return NULL;
//     if (!PyTuple_Check(coefficients_obj)) {
//         PyErr_Format(PyExc_TypeError,
//                      "The first argument must be a tuple representing the coefficients of the linear model !");
//         return NULL;
//     }
//     //std::cout << "activation_obj:" << activation_obj << std::endl;
//     //std::cout << "out_activation_obj:" << out_activation_obj << std::endl;


//     std::vector<double>* classes = tupleToVectorOneDimension(classes_obj);
//     std::vector<std::vector<std::vector<double>>>* coefficients = tupleToVectorThreeDimensions(coefficients_obj);
//     std::vector<std::vector<double>>* intercepts = tupleToVectorTwoDimensions(intercept_obj);

//     pyxai::ExplainerModelSpecific *explainer = new pyxai::ExplainerModelSpecific(classes, activation_obj, out_activation_obj, coefficients, intercepts);
//     return void_to_pyobject(explainer);
// }

// PyObject *new_linear_model(PyObject *self, PyObject *args) {
//     PyObject *classes_obj;
//     PyObject *coefficients_obj;
//     PyObject *intercept_obj;
//     if (!PyArg_ParseTuple(args, "OOO", &classes_obj, &coefficients_obj, &intercept_obj))
//         return NULL;
//     if (!PyTuple_Check(coefficients_obj)) {
//         PyErr_Format(PyExc_TypeError,
//                      "The first argument must be a tuple representing the coefficients of the linear model !");
//         return NULL;
//     }
//     //std::cout << "n_classes" << val << std::endl;
//     std::vector<double>* classes = tupleToVectorOneDimension(classes_obj);
//     std::vector<std::vector<double>>* coefficients = tupleToVectorTwoDimensions(coefficients_obj);
//     std::vector<double>* intercept = tupleToVectorOneDimension(intercept_obj);
    
//     pyxai::ExplainerModelSpecific *explainer = new pyxai::ExplainerModelSpecific(classes, coefficients, intercept);
//     return void_to_pyobject(explainer);
// }

// PyObject *model_predict(PyObject *self, PyObject *args) {
//     PyObject *class_obj;
//     PyObject *instance_obj;
//     if (!PyArg_ParseTuple(args, "OO", &class_obj, &instance_obj))
//         return NULL;
//     pyxai::ExplainerModelSpecific *explainer = (pyxai::ExplainerModelSpecific *) pyxai::pyobject_to_void(class_obj);
//     std::vector<double>* instance = tupleToVectorOneDimension(instance_obj);
//     int prediction = explainer->predict(*instance);
//     return PyLong_FromLong(prediction);
// }

// PyObject *check_abductive_model_specific_neural_network(PyObject *self, PyObject *args) {
//     PyObject *class_obj;
//     PyObject *instance_obj;
//     PyObject *upper_bounds_obj;
//     PyObject *lower_bounds_obj;
//     PyObject *reason_upper_bounds_obj;
//     PyObject *reason_lower_bounds_obj;
//     long n_samples;
//     if (!PyArg_ParseTuple(args, "OOOOOOL", &class_obj, &instance_obj, &upper_bounds_obj, &lower_bounds_obj, &reason_upper_bounds_obj, &reason_lower_bounds_obj, &n_samples))
//         return NULL;
//     std::cout << "n_samples:" << n_samples << std::endl;
    
//     pyxai::ExplainerModelSpecific *explainer = (pyxai::ExplainerModelSpecific *) pyxai::pyobject_to_void(class_obj);

//     std::vector<double>* instance = tupleToVectorOneDimension(instance_obj);
//     std::vector<double>* upper_bounds = tupleToVectorOneDimension(upper_bounds_obj);
//     std::vector<double>* lower_bounds = tupleToVectorOneDimension(lower_bounds_obj);

//     std::vector<double>* reason_upper_bounds = tupleToVectorOneDimension(reason_upper_bounds_obj);
//     std::vector<double>* reason_lower_bounds = tupleToVectorOneDimension(reason_lower_bounds_obj);

//     bool result = explainer->check_abductive_model_specific_neural_network(*instance, *upper_bounds, *lower_bounds, *reason_upper_bounds, *reason_lower_bounds, n_samples);
//     return result ? Py_True : Py_False;
// }

// PyObject *LRP_reason(PyObject *self, PyObject *args) {
//     PyObject *class_obj;
//     PyObject *instance_obj;
//     char *LRP_method;
//     if (!PyArg_ParseTuple(args, "OOs", &class_obj, &instance_obj, &LRP_method))
//         return NULL;
    
//     pyxai::ExplainerModelSpecific *explainer = (pyxai::ExplainerModelSpecific *) pyxai::pyobject_to_void(class_obj);
//     explainer->set_LRP(LRP_method);

//     std::vector<double>* instance = tupleToVectorOneDimension(instance_obj);
//     std::vector<double> reason;
    
//     if (explainer->model_type==NEURAL_NETWORK){
//         explainer->LRP_reason_neural_network(*instance, reason);
//         PyObject* py_tuple_reason = vectorToTuple_Float(&reason);    
//         return py_tuple_reason;

//     }else if (explainer->model_type==LINEAR_MODEL){
//         std::cout << "not implemented" << std::endl;
//         exit(0);
//     }
//     return NULL;

    

// }



// PyObject *model_specific_reason(PyObject *self, PyObject *args) {
//     PyObject *class_obj;
//     PyObject *instance_obj;
//     PyObject *upper_bounds_obj;
//     PyObject *lower_bounds_obj;
//     long n_iterations;
//     long time_limit;
//     char *LRP_method;
//     if (!PyArg_ParseTuple(args, "OOOOLLs", &class_obj, &instance_obj, &upper_bounds_obj, &lower_bounds_obj, &n_iterations, &time_limit, &LRP_method))
//         return NULL;
    
//     pyxai::ExplainerModelSpecific *explainer = (pyxai::ExplainerModelSpecific *) pyxai::pyobject_to_void(class_obj);

//     explainer->set_n_iterations(n_iterations);
//     explainer->set_time_limit(time_limit);
//     explainer->set_LRP(LRP_method);
    
//     std::vector<double>* instance = tupleToVectorOneDimension(instance_obj);
//     std::vector<double>* upper_bounds = tupleToVectorOneDimension(upper_bounds_obj);
//     std::vector<double>* lower_bounds = tupleToVectorOneDimension(lower_bounds_obj);
    
//     std::vector<double> result_upper_bounds;
//     std::vector<double> result_lower_bounds;

//     if (explainer->model_type==NEURAL_NETWORK){
//         explainer->model_specific_reason_neural_network(*instance, *upper_bounds, *lower_bounds, result_upper_bounds, result_lower_bounds);
//         PyObject* py_tuple_result_upper_bounds = vectorToTuple_Float(&result_upper_bounds);
//         PyObject* py_tuple_result_lower_bounds = vectorToTuple_Float(&result_lower_bounds);
//         PyObject* results = PyTuple_New(2);
//         PyTuple_SET_ITEM(results, 0, py_tuple_result_lower_bounds);
//         PyTuple_SET_ITEM(results, 1, py_tuple_result_upper_bounds);
//         return results;

//     }else if (explainer->model_type==LINEAR_MODEL){
//         std::vector<bool>* reason = explainer->model_specific_reason_linear_model(*instance, *upper_bounds, *lower_bounds);
//         if(reason->size() == 0)
//             return Py_None;

//         PyObject* tuple_reason = PyTuple_New(reason->size());
//         if (!tuple_reason) throw std::logic_error("Unable to allocate memory for Python tuple");
//         for (unsigned int i = 0; i < reason->size(); i++) {
//             PyObject *num = ((*reason)[i] == 0) ? Py_False : Py_True;
//             if (!num) {
//                 Py_DECREF(tuple_reason);
//                 throw std::logic_error("Unable to allocate memory for Python tuple");
//             }
//             PyTuple_SET_ITEM(tuple_reason, i, num);
//         }
//         return tuple_reason;
//     }
//     return NULL;
// }

// PyObject *minimal_model_specific_reasons(PyObject *self, PyObject *args) {
//     PyObject *class_obj;
//     PyObject *instance_obj;
//     PyObject *upper_bounds_obj;
//     PyObject *lower_bounds_obj;
//     long n_iterations;
//     long time_limit;
//     long method;
//     if (!PyArg_ParseTuple(args, "OOOOLLL", &class_obj, &instance_obj, &upper_bounds_obj, &lower_bounds_obj, &n_iterations, &time_limit, &method))
//        return NULL;
    
//     pyxai::ExplainerModelSpecific *explainer = (pyxai::ExplainerModelSpecific *) pyxai::pyobject_to_void(class_obj);

//     explainer->set_n_iterations(n_iterations);
//     explainer->set_time_limit(time_limit);
//     explainer->set_method(method);
    
//     std::vector<double>* instance = tupleToVectorOneDimension(instance_obj);
//     std::vector<double>* upper_bounds = tupleToVectorOneDimension(upper_bounds_obj);
//     std::vector<double>* lower_bounds = tupleToVectorOneDimension(lower_bounds_obj);
    
//     std::vector<double> result_upper_bounds;
//     std::vector<double> result_lower_bounds;
    

//     if (explainer->model_type==NEURAL_NETWORK){
//         explainer->minimal_model_specific_reasons_neural_network(*instance, *upper_bounds, *lower_bounds, result_upper_bounds, result_lower_bounds);
        
//         PyObject* py_tuple_result_upper_bounds = vectorToTuple_Float(&result_upper_bounds);
//         PyObject* py_tuple_result_lower_bounds = vectorToTuple_Float(&result_lower_bounds);
//         PyObject* results = PyTuple_New(2);
//         PyTuple_SET_ITEM(results, 0, py_tuple_result_lower_bounds);
//         PyTuple_SET_ITEM(results, 1, py_tuple_result_upper_bounds);
//         return results;
//     }else if (explainer->model_type==LINEAR_MODEL){
//         std::vector<std::vector<bool>*>* reasons = explainer->minimal_model_specific_reasons_linear_model(instance, upper_bounds, lower_bounds);
//         if(reasons->size() == 0)
//             return Py_None;


//         PyObject* tuple_reasons = PyTuple_New(reasons->size());
//         if (!tuple_reasons) throw std::logic_error("Unable to allocate memory for Python tuple");
//         for (unsigned int i = 0; i < reasons->size(); i++) {
//             std::vector<bool>* reason = (*reasons)[i];
//             PyObject* tuple_reason = PyTuple_New(reason->size());
            
//             for (unsigned int j = 0; j < reason->size(); j++) {
//                 PyObject *num = ((*reason)[j] == 0) ? Py_False : Py_True;
//                 if (!num) {
//                     Py_DECREF(tuple_reason);
//                     throw std::logic_error("Unable to allocate memory for Python tuple");
//                 }
//                 PyTuple_SET_ITEM(tuple_reason, j, num);
//             }    
            
            
//             PyTuple_SET_ITEM(tuple_reasons, i, tuple_reason);
//         }
//         return tuple_reasons;
    
//     }
//     return NULL;
// }



PyObject *new_classifier_RF(PyObject *self, PyObject *args) {
    long val;
    if (!PyArg_ParseTuple(args, "L", &val))
        PyErr_Format(PyExc_TypeError, "The argument must be a integer representing the number of classes");
    //std::cout << "n_classes" << val << std::endl;

    pyxai::TreeBasedExplainer *explainer = new pyxai::TreeBasedExplainer(val, pyxai::Classifier_RF);
    return void_to_pyobject(explainer);
}

PyObject *new_classifier_BT(PyObject *self, PyObject *args) {
    long val;
    if (!PyArg_ParseTuple(args, "L", &val))
        PyErr_Format(PyExc_TypeError, "The argument must be a integer representing the number of classes");
    //std::cout << "n_classes" << val << std::endl;

    pyxai::TreeBasedExplainer *explainer = new pyxai::TreeBasedExplainer(val, pyxai::Classifier_BT);
    return void_to_pyobject(explainer);
}

PyObject *new_regression_BT(PyObject *self, PyObject *args) {
    pyxai::TreeBasedExplainer *explainer = new pyxai::TreeBasedExplainer(0, pyxai::Regression_BT); // 0 because don't care number of classes
    return void_to_pyobject(explainer);
}

PyObject *new_rectifier(PyObject *self, PyObject *args) {
    pyxai::Rectifier *rectifier = new pyxai::Rectifier();
    return void_to_pyobject(rectifier);
}

static PyObject *rectifier_add_tree(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *tree_obj;
    if (!PyArg_ParseTuple(args, "OO", &class_obj, &tree_obj))
        return NULL;
    if (!PyTuple_Check(tree_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The second argument must be a tuple representing a raw tree and given by the python raw_tree() method !");
        return NULL;
    }
    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    rectifier->addTree(tree_obj);
    return Py_None;
}

static PyObject *rectifier_add_decision_rule(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *tree_obj;
    if (!PyArg_ParseTuple(args, "OO", &class_obj, &tree_obj))
        return NULL;
    if (!PyTuple_Check(tree_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The second argument must be a tuple representing a raw tree and given by the python raw_tree() method !");
        return NULL;
    }
    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    rectifier->addDecisionRule(tree_obj);
    return Py_None;
}

static PyObject *rectifier_improved_rectification(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *conditions_tuple;
    int label;

    if (!PyArg_ParseTuple(args, "OOi", &class_obj, &conditions_tuple, &label))
        return NULL;
    
    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    rectifier->improvedRectification(conditions_tuple, label); 
    return Py_None;
}


static PyObject *rectifier_disjoint_trees_decision_rule(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    rectifier->disjointTreesDecisionRule();
    return Py_None;
}

static PyObject *rectifier_concatenate_trees_decision_rule(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    rectifier->concatenateTreesDecisionRule();
    return Py_None;
}

static PyObject *rectifier_neg_decision_rules(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    rectifier->negatingDecisionRules();
    return Py_None;
}

static PyObject *rectifier_simplify_theory(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    rectifier->simplifyTheory();
    return Py_None;
}

static PyObject *rectifier_simplify_redundant(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    rectifier->simplifyRedundant();
    return Py_None;
}

static PyObject *rectifier_n_nodes(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    
    return Py_BuildValue("i", rectifier->nNodes());
}

static PyObject *rectifier_free(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    rectifier->free();
    return Py_None;
}

static PyObject *rectifier_get_tree(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    int id_tree_obj;

    if (!PyArg_ParseTuple(args, "Oi", &class_obj, &id_tree_obj))
        return NULL;

    
    // Get pointer to the class
    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    return rectifier->trees[id_tree_obj]->toTuple();
}

static PyObject *add_tree(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *tree_obj;

    if (!PyArg_ParseTuple(args, "OO", &class_obj, &tree_obj))
        return NULL;

    if (!PyTuple_Check(tree_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The second argument must be a tuple representing a raw tree and given by the python raw_tree() method !");
        return NULL;
    }
    // Get pointer to the class
    pyxai::TreeBasedExplainer *explainer = (pyxai::TreeBasedExplainer *) pyxai::pyobject_to_void(class_obj);
    explainer->addTree(tree_obj);
    return Py_None;
}


static PyObject *set_base_score(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    double bs;
    if (!PyArg_ParseTuple(args, "Od", &class_obj, &bs))
        return NULL;


    // Get pointer to the class
    pyxai::TreeBasedExplainer *explainer = (pyxai::TreeBasedExplainer *) pyxai::pyobject_to_void(class_obj);
    explainer->base_score = bs;
    return Py_None;
}

static PyObject *set_interval(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    double lower_bound, upper_bound;
    //std::cout << "add_tree" << std::endl;
    if (!PyArg_ParseTuple(args, "Odd", &class_obj, &lower_bound, &upper_bound))
        return NULL;


    // Get pointer to the class
    pyxai::TreeBasedExplainer *explainer = (pyxai::TreeBasedExplainer *) pyxai::pyobject_to_void(class_obj);
    explainer->set_interval(lower_bound, upper_bound);
    return Py_None;
}

static PyObject *set_excluded(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *vector_excluded_obj;
    if (!PyArg_ParseTuple(args, "OO", &class_obj, &vector_excluded_obj)) {
        return NULL;
    }
    if (!PyTuple_Check(vector_excluded_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The second argument must be a tuple representing the excluded features !");
        return NULL;
    }

    pyxai::TreeBasedExplainer *explainer = (pyxai::TreeBasedExplainer *) pyxai::pyobject_to_void(class_obj);
    explainer->excluded_features.clear();
    // Convert the vector of the instance

    Py_ssize_t size_obj = PyTuple_Size(vector_excluded_obj);
    for(int i = 0; i < size_obj; i++) {
        PyObject *value_obj = PyTuple_GetItem(vector_excluded_obj, i);
        explainer->excluded_features.push_back(PyLong_AsLong(value_obj));
    }

    return Py_None;
}

static PyObject *set_theory(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *vector_theory;
    if (!PyArg_ParseTuple(args, "OO", &class_obj, &vector_theory)) {
        return NULL;
    }
    if (!PyTuple_Check(vector_theory)) {
        PyErr_Format(PyExc_TypeError,
                     "The second argument must be a tuple reprenting the theory !");
        return NULL;
    }
    pyxai::TreeBasedExplainer *explainer = (pyxai::TreeBasedExplainer *) pyxai::pyobject_to_void(class_obj);

    // Convert the vector of the instance

    Py_ssize_t size_theory = PyTuple_Size(vector_theory);

    std::vector<std::vector<Lit> > clauses;
    int max = 0;
    for(int i = 0; i < size_theory; i++) {
        std::vector<Lit> c;
        PyObject *value_obj = PyTuple_GetItem(vector_theory, i);
        Py_ssize_t size_obj = PyTuple_Size(value_obj);
        if (size_obj != 2)
            throw std::logic_error("The clauses of the theory must be of size 2 (binary).");
        for(int i = 0; i < size_obj; i++) {
            long l = PyLong_AsLong(PyTuple_GetItem(value_obj, i));
            if(max < std::abs(l)) max = std::abs(l);
            c.push_back((l > 0) ? Lit::makeLit(l, false) : Lit::makeLit(-l, true));
        }
        clauses.push_back(c);
    }
    pyxai::Problem problem(clauses, max, std::cout, false);
    explainer->theory_propagator = new pyxai::Propagator(problem, false);
    for(pyxai::Tree *t : explainer->trees)
        t->propagator = explainer->theory_propagator;
    return Py_None;
}

static PyObject *rectifier_set_theory(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *vector_theory;
    if (!PyArg_ParseTuple(args, "OO", &class_obj, &vector_theory)) {
        return NULL;
    }
    if (!PyTuple_Check(vector_theory)) {
        PyErr_Format(PyExc_TypeError,
                     "The second argument must be a tuple reprenting the theory !");
        return NULL;
    }
    
    // Convert the vector of the instance

    Py_ssize_t size_theory = PyTuple_Size(vector_theory);

    std::vector<std::vector<Lit> > clauses;
    int max = 0;
    for(int i = 0; i < size_theory; i++) {
        std::vector<Lit> c;
        PyObject *value_obj = PyTuple_GetItem(vector_theory, i);
        Py_ssize_t size_obj = PyTuple_Size(value_obj);
        if (size_obj != 2)
            throw std::logic_error("The clauses of the theory must be of size 2 (binary).");
        for(int i = 0; i < size_obj; i++) {
            long l = PyLong_AsLong(PyTuple_GetItem(value_obj, i));
            if(max < std::abs(l)) max = std::abs(l);
            c.push_back((l > 0) ? Lit::makeLit(l, false) : Lit::makeLit(-l, true));
        }
        clauses.push_back(c);
    }
    pyxai::Problem problem(clauses, max, std::cout, false);

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyxai::pyobject_to_void(class_obj);
    rectifier->theory_propagator = new pyxai::Propagator(problem, false);
    for(pyxai::Tree *t : rectifier->trees)
        t->propagator = rectifier->theory_propagator;
    return Py_None;
}


static PyObject *compute_reason(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *vector_instance_obj;
    PyObject *vector_features_obj;
    long prediction;
    long n_iterations;
    long time_limit;
    char *features_expressivity;
    long seed;
    double theta;

    if (!PyArg_ParseTuple(args, "OOOLLLsLd", &class_obj, &vector_instance_obj, &vector_features_obj, &prediction, &n_iterations, &time_limit, &features_expressivity, &seed, &theta))
        return NULL;

    if (!PyTuple_Check(vector_instance_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The second argument must be a tuple reprenting the implicant !");
        return NULL;
    }

    if (!PyTuple_Check(vector_features_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The third argument must be a tuple representing the features !");
        return NULL;
    }
    std::vector<int> reason;
    std::vector<int> instance;
    std::vector<int> features;

    // Convert the vector of the instance 
    Py_ssize_t size_obj = PyTuple_Size(vector_instance_obj);
    for(int i = 0; i < size_obj; i++) {
        PyObject *value_obj = PyTuple_GetItem(vector_instance_obj, i);
        instance.push_back(PyLong_AsLong(value_obj));
    }

    // Convert the vector of the features 
    size_obj = PyTuple_Size(vector_features_obj);
    for(int i = 0; i < size_obj; i++) {
        PyObject *value_obj = PyTuple_GetItem(vector_features_obj, i);
        features.push_back(PyLong_AsLong(value_obj));
    }

    // Get pointer to the class
    pyxai::TreeBasedExplainer *explainer = (pyxai::TreeBasedExplainer *) pyxai::pyobject_to_void(class_obj);
    explainer->set_n_iterations(n_iterations);
    explainer->set_time_limit(time_limit);
    bool ret;
    if (strcmp(features_expressivity, "features") == 0)
      ret = explainer->compute_reason_features(instance, features, prediction, reason, theta);
    else
      ret = explainer->compute_reason_conditions(instance, prediction, reason, seed, theta);

    if(ret == false)
        return Py_None;
        
    return pyxai::vectorToTuple_Int(reason);
}


// See https://gist.github.com/physacco/2e1b52415f3a964ad2a542a99bebed8f

// Method definition object for this extension, these arguments mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//          accepting arguments, accepting keyword arguments, being a
//          class method, or being a static method of a class.
// ml_doc:  Contents of this method's docstring

static PyMethodDef module_methods[] = {

    
        //{"new_neural_network", new_neural_network, METH_VARARGS, "Create a neural network model."},
        //{"new_linear_model", new_linear_model, METH_VARARGS, "Create a linear model."},
        {"new_classifier_BT", new_classifier_BT, METH_VARARGS, "Create a Classifier_BT explainer."},
        {"new_classifier_RF", new_classifier_RF, METH_VARARGS, "Create a Classifier_RF explainer."},
        {"new_regression_BT", new_regression_BT, METH_VARARGS, "Create a regression BT explainer."},
        {"new_rectifier", new_rectifier, METH_VARARGS, "Create a rectifier."},
        
        //{"minimal_model_specific_reasons", minimal_model_specific_reasons, METH_VARARGS, "Compute several minimal model specific reasons."},
        //{"model_specific_reason", model_specific_reason, METH_VARARGS, "Return one random model specific reason."},
        //{"LRP_reason", LRP_reason, METH_VARARGS, "Return the LRP explanation."},
        
        //{"check_abductive_model_specific_neural_network", check_abductive_model_specific_neural_network, METH_VARARGS, "Check if the reason is abductive: All instances that can be derived from reason must have the same classification."},
        
        //{"model_predict", model_predict, METH_VARARGS, "Return the prediction."},
        
        
        {"rectifier_add_tree", rectifier_add_tree, METH_VARARGS, "Add a tree in the rectifier."},
        {"rectifier_add_decision_rule", rectifier_add_decision_rule, METH_VARARGS, "Set tree."},
        {"rectifier_improved_rectification", rectifier_improved_rectification, METH_VARARGS, "Improved Rectification."},
        {"rectifier_neg_decision_rules", rectifier_neg_decision_rules, METH_VARARGS, "Negating tree."},
        {"rectifier_disjoint_trees_decision_rule", rectifier_disjoint_trees_decision_rule, METH_VARARGS, "rectifier_disjoint_trees_decision_rule."},
        {"rectifier_concatenate_trees_decision_rule", rectifier_concatenate_trees_decision_rule, METH_VARARGS, "rectifier_concatenate_trees_decision_rule tree."},
        {"rectifier_get_tree", rectifier_get_tree, METH_VARARGS, "rectifier_get_tree tree."},
        {"rectifier_set_theory", rectifier_set_theory, METH_VARARGS, "rectifier_set_theory tree."},
        {"rectifier_simplify_theory", rectifier_simplify_theory, METH_VARARGS, "rectifier_simplify_theory tree."},
        {"rectifier_n_nodes", rectifier_n_nodes, METH_VARARGS, "rectifier_n_nodes tree."},
        {"rectifier_simplify_redundant", rectifier_simplify_redundant, METH_VARARGS, "rectifier_simplify_redundant tree."},
        {"rectifier_free", rectifier_free, METH_VARARGS, "rectif tree."},
        {"add_tree",          add_tree,          METH_VARARGS, "Add a tree."},
        {"set_excluded",      set_excluded,      METH_VARARGS, "Set excluded features"},
        {"set_theory",        set_theory,        METH_VARARGS, "Set the theory"},
        {"compute_reason",    compute_reason,    METH_VARARGS, "Compute a reason"},
        {"set_interval",      set_interval,      METH_VARARGS, "Set the interval (useful for regression)"},
        {"set_base_score",      set_base_score,  METH_VARARGS, "Set the base score (useful for regression)"},
        {NULL,             NULL,                 0,            NULL}
};


// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for it's method definitions
static struct PyModuleDef module_definition = {
        PyModuleDef_HEAD_INIT,
        "c_explainer",
        "Explainer in C++ in order to improve performances.",
        -1,
        module_methods
};

// Module initialization
// Python calls this function when importing your extension. It is important
// that this function is named PyInit_[[your_module_name]] exactly, and matches
// the name keyword argument in setup.py's setup() call.
PyMODINIT_FUNC PyInit_c_explainer(void) {
    //Py_Initialize();
    return PyModule_Create(&module_definition);
}

#endif //CPP_CODE_WRAPPER_H
