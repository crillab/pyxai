//
// Created by audemard on 22/04/2022.
//

#ifndef CPP_CODE_BT_WRAPPER_H
#define CPP_CODE_BT_WRAPPER_H

#include<Python.h>
#include "Node.h"
#include "Explainer.h"
#include "Rectifier.h"

using namespace pyxai;
static PyObject* vectorToTuple_Int(const std::vector<int> &data) {
    PyObject* tuple = PyTuple_New( data.size() );
    if (!tuple) throw std::logic_error("Unable to allocate memory for Python tuple");
    for (unsigned int i = 0; i < data.size(); i++) {
        PyObject *num = PyLong_FromLong(data[i]);
        if (!num) {
            Py_DECREF(tuple);
            throw std::logic_error("Unable to allocate memory for Python tuple");
        }
        PyTuple_SET_ITEM(tuple, i, num);
    }
    return tuple;
}


static PyObject *void_to_pyobject(void *ptr) {
    return PyCapsule_New(ptr, NULL, NULL);
}


static void *pyobject_to_void(PyObject *obj) {
    return PyCapsule_GetPointer(obj, NULL);
}


PyObject *new_classifier_RF(PyObject *self, PyObject *args) {
    long val;
    if (!PyArg_ParseTuple(args, "L", &val))
        PyErr_Format(PyExc_TypeError, "The argument must be a integer representing the number of classes");
    //std::cout << "n_classes" << val << std::endl;

    pyxai::Explainer *explainer = new pyxai::Explainer(val, pyxai::Classifier_RF);
    return void_to_pyobject(explainer);
}

PyObject *new_classifier_BT(PyObject *self, PyObject *args) {
    long val;
    if (!PyArg_ParseTuple(args, "L", &val))
        PyErr_Format(PyExc_TypeError, "The argument must be a integer representing the number of classes");
    //std::cout << "n_classes" << val << std::endl;

    pyxai::Explainer *explainer = new pyxai::Explainer(val, pyxai::Classifier_BT);
    return void_to_pyobject(explainer);
}

PyObject *new_regression_BT(PyObject *self, PyObject *args) {
    pyxai::Explainer *explainer = new pyxai::Explainer(0, pyxai::Regression_BT); // 0 because don't care number of classes
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
    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
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
    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
    rectifier->addDecisionRule(tree_obj);
    return Py_None;
}

static PyObject *rectifier_improved_rectification(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *conditions_tuple;
    int label;

    if (!PyArg_ParseTuple(args, "OOi", &class_obj, &conditions_tuple, &label))
        return NULL;
    
    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
    rectifier->improvedRectification(conditions_tuple, label); 
    return Py_None;
}


static PyObject *rectifier_disjoint_trees_decision_rule(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
    rectifier->disjointTreesDecisionRule();
    return Py_None;
}

static PyObject *rectifier_concatenate_trees_decision_rule(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
    rectifier->concatenateTreesDecisionRule();
    return Py_None;
}

static PyObject *rectifier_neg_decision_rules(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
    rectifier->negatingDecisionRules();
    return Py_None;
}

static PyObject *rectifier_simplify_theory(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
    rectifier->simplifyTheory();
    return Py_None;
}

static PyObject *rectifier_simplify_redundant(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
    rectifier->simplifyRedundant();
    return Py_None;
}

static PyObject *rectifier_n_nodes(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
    
    return Py_BuildValue("i", rectifier->nNodes());
}

static PyObject *rectifier_free(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    if (!PyArg_ParseTuple(args, "O", &class_obj))
        return NULL;

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
    rectifier->free();
    return Py_None;
}

static PyObject *rectifier_get_tree(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    int id_tree_obj;

    if (!PyArg_ParseTuple(args, "Oi", &class_obj, &id_tree_obj))
        return NULL;

    
    // Get pointer to the class
    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
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
    pyxai::Explainer *explainer = (pyxai::Explainer *) pyobject_to_void(class_obj);
    explainer->addTree(tree_obj);
    return Py_None;
}


static PyObject *set_base_score(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    double bs;
    if (!PyArg_ParseTuple(args, "Od", &class_obj, &bs))
        return NULL;


    // Get pointer to the class
    pyxai::Explainer *explainer = (pyxai::Explainer *) pyobject_to_void(class_obj);
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
    pyxai::Explainer *explainer = (pyxai::Explainer *) pyobject_to_void(class_obj);
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

    pyxai::Explainer *explainer = (pyxai::Explainer *) pyobject_to_void(class_obj);
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
    pyxai::Explainer *explainer = (pyxai::Explainer *) pyobject_to_void(class_obj);

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

    pyxai::Rectifier *rectifier = (pyxai::Rectifier *) pyobject_to_void(class_obj);
    rectifier->theory_propagator = new pyxai::Propagator(problem, false);
    for(pyxai::Tree *t : rectifier->trees)
        t->propagator = rectifier->theory_propagator;
    return Py_None;
}


static PyObject *compute_reason(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *vector_instance_obj;
    PyObject *vector_features_obj;
    PyObject *vector_weights_obj;
    
    long prediction;
    long n_iterations;
    long time_limit;
    long features_expressivity;
    long seed;
    double theta;

    if (!PyArg_ParseTuple(args, "OOOOLLLLLd", &class_obj, &vector_instance_obj, &vector_features_obj, &vector_weights_obj, &prediction, &n_iterations, &time_limit, &features_expressivity, &seed, &theta))
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

    if (!PyTuple_Check(vector_weights_obj)) {
        PyErr_Format(PyExc_TypeError,
                     "The argument 4 must be a tuple representing the features !");
        return NULL;
    }

    std::vector<int> reason;
    std::vector<int> instance;
    std::vector<int> features;
    std::vector<int> weights;

    // Convert the vector of the instance 
    Py_ssize_t size_obj = PyTuple_Size(vector_weights_obj);
    if (size_obj != -1){ // -1 when the tuple is empty
        for(int i = 0; i < size_obj; i++) {
            PyObject *value_obj = PyTuple_GetItem(vector_weights_obj, i);
            weights.push_back(PyLong_AsLong(value_obj));
        }
    }
    

    // Convert the vector of the instance 
    size_obj = PyTuple_Size(vector_instance_obj);
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
    pyxai::Explainer *explainer = (pyxai::Explainer *) pyobject_to_void(class_obj);
    explainer->set_n_iterations(n_iterations);
    explainer->set_time_limit(time_limit);
    bool ret;
    if (features_expressivity == 1)
      ret = explainer->compute_reason_features(instance, features, prediction, reason, theta);
    else
      ret = explainer->compute_reason_conditions(instance, weights, prediction, reason, seed, theta);

    if(ret == false)
        return Py_None;
    
    return vectorToTuple_Int(reason);
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
        {"new_classifier_BT", new_classifier_BT, METH_VARARGS, "Create a Classifier_BT explainer."},
        {"new_classifier_RF", new_classifier_RF, METH_VARARGS, "Create a Classifier_RF explainer."},
        {"new_regression_BT", new_regression_BT, METH_VARARGS, "Create a regression BT explainer."},
        {"new_rectifier", new_rectifier, METH_VARARGS, "Create a rectifier."},
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

#endif //CPP_CODE_BT_WRAPPER_H
