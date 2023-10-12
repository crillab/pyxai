//
// Created by audemard on 22/04/2022.
//

#ifndef CPP_CODE_BT_WRAPPER_H
#define CPP_CODE_BT_WRAPPER_H

#include<Python.h>
#include "Node.h"
#include "Explainer.h"

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


static PyObject *compute_reason(PyObject *self, PyObject *args) {
    PyObject *class_obj;
    PyObject *vector_instance_obj;
    PyObject *vector_features_obj;
    long prediction;
    long n_iterations;
    long time_limit;
    long features_expressivity;
    long seed;
    if (!PyArg_ParseTuple(args, "OOOLLLLL", &class_obj, &vector_instance_obj, &vector_features_obj, &prediction, &n_iterations, &time_limit, &features_expressivity, &seed))
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
    pyxai::Explainer *explainer = (pyxai::Explainer *) pyobject_to_void(class_obj);
    explainer->set_n_iterations(n_iterations);
    explainer->set_time_limit(time_limit);
    bool ret;
    if (features_expressivity == 1)
      ret = explainer->compute_reason_features(instance, features, prediction, reason);
    else
      ret = explainer->compute_reason_conditions(instance, prediction, reason, seed);

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
