
#ifndef CPP_CODE_PYTHON_TO_C_MEMORY_H
#define CPP_CODE_PYTHON_TO_C_MEMORY_H

#include <Python.h>
#include "NeuralNetwork.h"

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

    
    static std::vector<Layer*>* PyObject_to_Layers(PyObject *object) {
        std::vector<Layer*>* layers = new std::vector<Layer*>();
        Py_ssize_t size_obj = PyTuple_Size(object);
        if (!PyTuple_Check(object)) {
            throw std::logic_error("It is not a tuple");
        }
        for (unsigned int i = 0; i < (unsigned int)size_obj; i++) {
            PyObject* layer_obj = PyTuple_GetItem(object, i);
            if (!PyTuple_Check(layer_obj)) {
                throw std::logic_error("It is not a tuple");
            }
            const char* layer_type;
            PyObject* layer_data_obj;
            if (!PyArg_ParseTuple(layer_obj, "sO", &layer_type, &layer_data_obj))
                throw std::logic_error("It is not a tuple or this tuple is not well formatted (string, tuple)");
            //std::cout << "layer_type:" << layer_type << std::endl;
            if (strcmp(layer_type, "LinearLayer") == 0) {
                std::vector<std::vector<double>>* coefficients = tupleToVectorTwoDimensions(PyTuple_GetItem(layer_data_obj,0));
                std::vector<double>* intercepts = tupleToVectorOneDimension(PyTuple_GetItem(layer_data_obj,1));
                Layer* layer = new LinearLayer(coefficients, intercepts);
                layers->push_back(layer);
            } else if (strcmp(layer_type, "ActivationLayer") == 0) {
                const char* activation_name;
                if (!PyArg_ParseTuple(layer_data_obj, "s", &activation_name))
                    throw std::logic_error("It is not a tuple or this tuple is not well formatted (string)");
                Layer* layer = new ActivationLayer(str_to_activation_type(activation_name));
                layers->push_back(layer);
            } else if (strcmp(layer_type, "FunctionLayer") == 0) {
                const char* function_name;
                if (!PyArg_ParseTuple(layer_data_obj, "s", &function_name))
                    throw std::logic_error("It is not a tuple or this tuple is not well formatted (string)");
                Layer* layer = new FunctionLayer(str_to_fonction_name(function_name));
                layers->push_back(layer);
            } else {
                throw std::logic_error("Unknown layer type: " + std::string(layer_type));
            }

            //Py_ssize_t size_layer_obj = PyTuple_Size(layer_obj);
            //const char* layer_type = toChars(PyTuple_GetItem(layer_obj, 1));
            //std::cout << "layer_type:" << layer_type << std::endl;
            //for (unsigned int j = 0;j < (unsigned int)size_layer_obj; j++) {
            //    
            //}
                

            //layers.push_back(layer);
        }
        return layers;
    }


}
#endif //CPP_CODE_PYTHON_TO_C_MEMORY_H
