
#ifndef CPP_CODE_PYTHON_TO_C_MEMORY_H
#define CPP_CODE_PYTHON_TO_C_MEMORY_H

#include <Python.h>

#include <sstream>
#include <limits>
#include <iostream>
#include <string>
#include <vector>

// TODO: Use PyErr_Occurred() to check for errors in PyArg_ParseTuple and other Python C API functions. 

namespace pyxai {

    static const char* toChars(PyObject * _value)
    {
        return PyBytes_AS_STRING(_value);
    }

    static  PyObject* convert_PyDouble(double value){
        return PyFloat_FromDouble(value);
        //return PyFloat_FromDouble(std::round(value / PRECISION) * PRECISION);
    }

    
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


    

    static PyObject* vectorToTuple_Float(const std::vector<double>* data) {
        PyObject* tuple = PyTuple_New( data->size() );
        if (!tuple) throw std::logic_error("Unable to allocate memory for Python tuple");
        for (unsigned int i = 0; i < data->size(); i++) {
            //std::cout << "ess:" << (*data)[i] << std::endl;
            PyObject *num = convert_PyDouble((*data)[i]);
            if (!num) {
                Py_DECREF(tuple);
                throw std::logic_error("Unable to allocate memory for Python tuple");
            }
            PyTuple_SET_ITEM(tuple, i, num);
        }
        return tuple;
    }

    

    /*
    * One dimension
    */
    static std::vector<double>* tupleToVectorOneDimension(PyObject *object) {
        std::vector<double>* vec = new std::vector<double>();
        Py_ssize_t size_obj = PyTuple_Size(object);
        if (!PyTuple_Check(object)) {
            throw std::logic_error("It is not a tuple");
        }
        for (unsigned int i = 0; i < (unsigned int)size_obj; i++) {
            PyObject *value = PyTuple_GetItem(object, i);
            vec->push_back(PyFloat_AsDouble(value));
        }
        return vec;
    }

    /*
    * Two dimensions
    */
    static std::vector<std::vector<double>>* tupleToVectorTwoDimensions(PyObject *object) {
        std::vector<std::vector<double>>* vec = new std::vector<std::vector<double>>();
        Py_ssize_t size_obj = PyTuple_Size(object);
        if (!PyTuple_Check(object)) {
            throw std::logic_error("It is not a tuple");
        }
        for (unsigned int i = 0; i < (unsigned int)size_obj; i++) {
            std::vector<double> vec_inside;
            PyObject *object_inside = PyTuple_GetItem(object, i);
            if (!PyTuple_Check(object_inside)) {
                throw std::logic_error("It is not a tuple");
            }
            Py_ssize_t size_obj2 = PyTuple_Size(object_inside);
            for (unsigned int j = 0; j < (unsigned int)size_obj2; j++) {
                PyObject *value = PyTuple_GetItem(object_inside, j);
                vec_inside.push_back(PyFloat_AsDouble(value));
            }
            vec->push_back(vec_inside);
            
        }
        return vec;
    }

    /*
    * Three dimensions
    */
    static std::vector<std::vector<std::vector<double>>>* tupleToVectorThreeDimensions(PyObject *object) {
        std::vector<std::vector<std::vector<double>>>* vec = new std::vector<std::vector<std::vector<double>>>();
        Py_ssize_t size_obj = PyTuple_Size(object);
        if (!PyTuple_Check(object)) {
            throw std::logic_error("It is not a tuple 1");
        }
        for (unsigned int i = 0; i < (unsigned int)size_obj; i++) {
            std::vector<std::vector<double>> vec_inside;
            
            PyObject *object_inside = PyTuple_GetItem(object, i);
            if (!PyTuple_Check(object_inside)) {
                throw std::logic_error("It is not a tuple 2");
            }
            Py_ssize_t size_objj = PyTuple_Size(object_inside);

            for (unsigned int j = 0; j < (unsigned int)size_objj; j++) {
            
                std::vector<double> vec_values;
                PyObject *object_values = PyTuple_GetItem(object_inside, j);
                if (!PyTuple_Check(object_values)) {
                    throw std::logic_error("It is not a tuple 3");
                }
                
                Py_ssize_t size_obj = PyTuple_Size(object_values);
                for (unsigned int k = 0; k < (unsigned int)size_obj; k++) {
                    PyObject *value = PyTuple_GetItem(object_values, k);
                    vec_values.push_back(PyFloat_AsDouble(value));
                }
                vec_inside.push_back(vec_values);
            }
            vec->push_back(vec_inside);    
        }
        return vec;
    }


    static int countDimensions(PyObject* object) {
        int dimensions = 0;
        PyObject* current_level = object;
        while (PyTuple_Check(current_level)) {
            dimensions++;
            if (PyTuple_Size(current_level) == 0) {
                break; // Avoid infinite loop on empty tuples
            }
            current_level = PyTuple_GetItem(current_level, 0);
        }
        return dimensions;
    }

    static void* tupleToVector(PyObject* object, int n_dimensions) {
        void* result = nullptr;
        if (n_dimensions == 1) {
            result = (void*) tupleToVectorOneDimension(object);
        } else if (n_dimensions == 2) {
            result = (void*) tupleToVectorTwoDimensions(object);
        } else if (n_dimensions == 3) {
            result = (void*) tupleToVectorThreeDimensions(object);
        } else {
            throw std::logic_error("Unsupported number of dimensions: " + std::to_string(n_dimensions));
        }
        return result;
    }


    static PyObject *void_to_pyobject(void *ptr) {
        return PyCapsule_New(ptr, NULL, NULL);
    }


    static void *pyobject_to_void(PyObject *obj) {
        return PyCapsule_GetPointer(obj, NULL);
    }

    

}
#endif //CPP_CODE_PYTHON_TO_C_MEMORY_H
