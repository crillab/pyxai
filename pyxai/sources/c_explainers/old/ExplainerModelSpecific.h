
#ifndef CPP_CODE_EXPLAINER_LINEAR_H
#define CPP_CODE_EXPLAINER_LINEAR_H

#include <Python.h>
#include <vector>
#include <iostream>
#include <limits>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>

#include "constants.h"

static const bool VERBOSE=1;
static const bool PRECISION_INTERVAL=6;


namespace pyxai {

    typedef enum {LINEAR_MODEL, NEURAL_NETWORK} ModelType;
    typedef enum {METHOD1, METHOD2} Method;
    
    typedef enum {NONE_LRP, NORMAL_DOWN, NORMAL_UP} LRPType;
    
    typedef enum {LT, LE, GT, GE} InequationSign; //LT: <, LE: <=, GT: >, GE >=
   
    class LinearModel{
        public:
            std::vector<std::vector<double>>* coefficients;
            std::vector<double>* intercepts;

            LinearModel() :

                        coefficients(nullptr), 
                        intercepts(nullptr) {} 
        
            LinearModel(std::vector<std::vector<double>>* _coefficients, 
                        std::vector<double>* _intercepts) :

                        coefficients(_coefficients), 
                        intercepts(_intercepts) {} 
        
            inline double _predict(std::vector<double>& _instance, std::vector<double>& _coefficients, double intercept){
                double prediction = 0;
                for (unsigned int i = 0; i < _instance.size();i++){
                    prediction += (_instance[i] * _coefficients[i]);
                }
                return prediction + intercept;
            }

            inline double _worst_best_instance_predict(std::vector<double>& _instance, std::vector<double>& _upper_bounds, std::vector<double>& _lower_bounds, std::vector<double>& _coefficients, double _intercept, unsigned int limit, std::vector<bool>& reason){
                double prediction = 0;
                //std::cout << "limit:" << limit << std::endl;
                for (unsigned int i = 0; i < _instance.size();i++){
                    double value = 0;
                    if (i <= limit && _coefficients[i] >= 0 && reason[i] == false){
                        //lower bound value
                        value = _lower_bounds[i];
                    }else if (i <= limit && _coefficients[i] < 0 && reason[i] == false){
                        //upper bound value
                        value = _upper_bounds[i];
                    }else {
                        value = _instance[i];
                    }
                    //std::cout << "value:" << value << std::endl;
                    prediction += (value * _coefficients[i]);
                }
                return prediction + _intercept;
            }

            inline int predict_class(std::vector<double>& _instance, std::vector<double>& _predictions, std::vector<double>& classes){
                if (_predictions.size() == 1){
                    // Only one model => Binary case
                    return (_predictions[0] >= 0) ? 1: 0;
                }else{
                    // Several models => Multi-classes case
                    double max = -std::numeric_limits<double>::infinity();
                    int position_max = 0;
                    for (unsigned int j = 0; j < _predictions.size();j++){
                        if (_predictions[j] > max){
                            max = _predictions[j];
                            position_max = j;
                        }
                    }
                    return classes[position_max];
                }
            }

            inline int predict(std::vector<double>& _instance, std::vector<double>& classes){
                unsigned int n_models = classes.size() == 2? 1: classes.size();
                std::vector<double> predictions(n_models, 0);
                for (unsigned int j = 0; j < n_models;j++){
                        predictions[j] = _predict(_instance, (*coefficients)[j], (*intercepts)[j]);
                }
                return predict_class(_instance, predictions, classes);
            }
    };

    class NeuralNetwork{
        public:
            ActivationType activation;
            ActivationType out_activation;
            LRPType LRP_method;

            std::vector<std::vector<std::vector<double>>>* coefficients;
            std::vector<std::vector<double>>* intercepts;
            
            NeuralNetwork(): 
                        activation(NONE_ACTIVATION),
                        out_activation(NONE_ACTIVATION),
                        LRP_method(NONE_LRP),
                        coefficients(nullptr), 
                        intercepts(nullptr) {}

            NeuralNetwork(char* _activation, char* _out_activation, std::vector<std::vector<std::vector<double>>>* _coefficients, std::vector<std::vector<double>>* _intercepts) : 
                        activation(get_activation(_activation)),
                        out_activation(get_activation(_out_activation)),
                        LRP_method(NONE_LRP),
                        coefficients(_coefficients), 
                        intercepts(_intercepts) {} 

            inline void set_LRP(char* str_LRP){
                if (VERBOSE){
                    std::cout << "LRP:" << str_LRP << std::endl;
                }
                LRP_method = get_LRP_from_string(str_LRP);
            }

            inline LRPType get_LRP_from_string(char* str_LRP){
                if (strcmp(str_LRP, "normal_down") == 0) return LRPType::NORMAL_DOWN;
                else if (strcmp(str_LRP, "normal_up") == 0) return LRPType::NORMAL_UP;
                else if (strcmp(str_LRP, "none") == 0) return LRPType::NONE_LRP;
                else {
                    std::cout << "get_LRP:" << str_LRP << " error." << std::endl;
                    exit(0);
                }
                return NONE_LRP;
            }

            inline LRPType get_LRP(){
                return LRP_method;
            }

            inline ActivationType get_activation(char* str_activation){
                //std::cout << "str_activation:" << str_activation << std::endl;
                if (strcmp(str_activation, "softmax") == 0) return ActivationType::SOFTMAX;
                else if (strcmp(str_activation, "logistic") == 0) return ActivationType::LOGISTIC;
                else if (strcmp(str_activation, "relu") == 0) return ActivationType::RELU;
                else if (strcmp(str_activation, "heaviside") == 0) return ActivationType::HEAVISIDE;
                else if (strcmp(str_activation, "none") == 0) return ActivationType::NONE_ACTIVATION;
                else {
                    std::cout << "get_activation:" << str_activation << " error." << std::endl;
                    exit(0);
                }
                return NONE_ACTIVATION;
            }

            inline void apply_activation(ActivationType _activation, std::vector<double>& values){
                if (_activation == SOFTMAX) softmax(values);
                else if (_activation == LOGISTIC) logistic(values);
                else if (_activation == RELU) relu(values);
                else if (_activation == HEAVISIDE) heaviside(values);
            }

            inline void apply_activation_upper_lower(ActivationType _activation, std::vector<double>& upper_bounds, std::vector<double>& lower_bounds){
                if (_activation == NONE_ACTIVATION){
                }else {
                    apply_activation(_activation, upper_bounds);
                    apply_activation(_activation, lower_bounds);
                }
            }


            inline void softmax(std::vector<double>& values){
                double sum = 0;
                for(double value:values){sum=sum+exp(value);}
                for (unsigned int i = 0; i < values.size(); i++){values[i] = exp(values[i])/sum;}
            }

            inline void logistic(std::vector<double>& values){
                for (unsigned int i = 0; i < values.size(); i++){
                    values[i] = 1/(1+exp(-values[i]));
                }
            }

            inline void relu(std::vector<double>& values){
                for (unsigned int i = 0; i < values.size(); i++){values[i] = std::max(double(0), values[i]);}
            }

            inline void heaviside(std::vector<double>& values){
                for (unsigned int i = 0; i < values.size(); i++){values[i] = values[i] >= 0 ? double(1) : double(0);}
            }

            inline double _predict(std::vector<double>& _instance, std::vector<double>& _coefficients, double intercept){
                return 0;
            }

            inline double _worst_best_instance_predict(std::vector<double>& _instance, std::vector<double>& _upper_bounds, std::vector<double>& _lower_bounds, std::vector<double>& _coefficients, double _intercept, unsigned int limit, std::vector<bool>& reason){
                return 0;
            }

            inline std::vector<std::vector<double>>* matrix_product_intercept_upper_lower(std::vector<std::vector<double>>& _coefficients, std::vector<double> _intercepts, std::vector<double>& upper_bounds, std::vector<double>& lower_bounds){
                // for each line of weights
                std::vector<std::vector<double>>* results = new std::vector<std::vector<double>>();
                std::vector<double> results_lower; 
                std::vector<double> results_upper; 

                unsigned int n_neurones_previous_layer = _coefficients.size();
                unsigned int n_neurones_current_layer = _coefficients[0].size();
                
                for (unsigned int i = 0; i < n_neurones_current_layer; i++){
                    double sum_lower = 0;
                    double sum_upper = 0;
                    
                    for (unsigned int j = 0; j < n_neurones_previous_layer; j++){
                        if (_coefficients[j][i] >= 0){
                            sum_lower += (lower_bounds[j] * _coefficients[j][i]);
                            sum_upper += (upper_bounds[j] * _coefficients[j][i]);
                        }else{
                            sum_lower += (upper_bounds[j] * _coefficients[j][i]);
                            sum_upper += (lower_bounds[j] * _coefficients[j][i]);
                        }

                    }
                    //std::cout << "sum_lower:" << std::endl;
                    results_lower.push_back(sum_lower + _intercepts[i]);
                    results_upper.push_back(sum_upper + _intercepts[i]);

                }
                results->push_back(results_lower);
                results->push_back(results_upper);
                return results;
            }

            inline void predict_matrix_product_intercept(std::vector<double>& current_values, std::vector<std::vector<double>>& _coefficients, std::vector<double> _intercepts){
                // for each line of weights
                std::vector<double> results; 

                unsigned int n_neurones_previous_layer = _coefficients.size();
                unsigned int n_neurones_current_layer = _coefficients[0].size();
                
                for (unsigned int i = 0; i < n_neurones_current_layer; i++){
                    double sum = 0;
                    for (unsigned int j = 0; j < n_neurones_previous_layer; j++){
                        sum += (current_values[j] * _coefficients[j][i]);
                    }
                    results.push_back(sum + _intercepts[i]);
                }

                current_values.clear();
                for (unsigned int i = 0; i < n_neurones_current_layer; i++){
                    current_values.push_back(results[i]);
                }
            }


            inline int predict_class(std::vector<double>& _instance, std::vector<double>& current_values, std::vector<double>& classes){
                if (current_values.size() == 1){
                    // Only one model => Binary case
                    //std::cout << "predict_class[0]:" << current_values[0] << std::endl;
                    int position = current_values[0] >= 1-current_values[0]? 1 :0; //std::max(1-current_values[0], current_values[0]);
                    //std::cout << "classes[position]:" << classes[position] << std::endl;
                    
                    return classes[position];
                }else{
                    // Several models => Multi-classes case
                    double max = -std::numeric_limits<double>::infinity();
                    int position_max = 0;
                    for (unsigned int j = 0; j < current_values.size();j++){
                        if (current_values[j] > max){
                            max = current_values[j];
                            position_max = j;
                        }
                    }
                    return classes[position_max];
                }
            }

            inline void display_instance(std::vector<double>& _instance){
                for (unsigned int i = 0; i< _instance.size(); i++){
                    std::cout << std::fixed << _instance[i] << " ";
                }
                std::cout << std::endl;
            }

            inline int predict(std::vector<double>& _instance, std::vector<double>& classes){
                //Perform a forward pass on the network by computing the values
                //of the neurons in the hidden layers
                std::vector<double> current_values(_instance);
                unsigned int n_layers = coefficients->size();
                
                for (unsigned int i = 0; i < n_layers;i++){
                    // The ith element of the list holds the values of the ith layer.
                    // display_instance(current_values);
                    predict_matrix_product_intercept(current_values, (*coefficients)[i], (*intercepts)[i]);
                    // std::cout << "layer " << i << ": ";
                    // display_instance(current_values);
                    if (i != coefficients->size()-1){
                        // Apply the activation function except on the last layer
                        apply_activation(activation, current_values);
                    }
                    //display_instance(current_values);
                }
                // Apply out activation function on the output layer: this one depends on the problem: binary or multi class
                //std::cout << "before predict_class[0]:" << current_values[0] << std::endl;
                apply_activation(out_activation, current_values);

                return predict_class(_instance, current_values, classes);
            }

            inline void _compute_LRP(std::vector<double>& current_values_LRP, std::vector<double>& previous_values_LRP, std::vector<std::vector<double>>& _coefficients, std::vector<double> _intercepts, std::vector<double>& instance_values){
                unsigned int n_neurones_previous_layer = _coefficients.size();
                unsigned int n_neurones_current_layer = _coefficients[0].size();
                if (VERBOSE){
                    display_instance(instance_values);
                    std::cout << "n_neurones_previous_layer:" << n_neurones_previous_layer << std::endl;
                    std::cout << "n_neurones_current_layer:" << n_neurones_current_layer << std::endl;
                }
                for (unsigned int j = 0; j < n_neurones_previous_layer; j++){
                    current_values_LRP.push_back(0);
                }
                for (unsigned int i = 0; i < n_neurones_current_layer; i++){
                    double sum = 0;
                    for (unsigned int j = 0; j < n_neurones_previous_layer; j++){
                        sum += (instance_values[j] * _coefficients[j][i]);
                    }  
                    // Add the intercept 

                    if (VERBOSE){
                        std::cout << "sum: " << sum << std::endl;
                        std::cout << "+intercept: " << _intercepts[i] << std::endl;
                    
                    }
                    //sum += _intercepts[i];
                    sum += 0.0001;
                    for (unsigned int j = 0; j < n_neurones_previous_layer; j++){
                        double alone = instance_values[j] * _coefficients[j][i];
                        if (VERBOSE){
                            std::cout << "for " << j << " : (" << alone << " / " << sum << " ) * " << previous_values_LRP[i] << std::endl;
                        }
                        current_values_LRP[j] += ((alone/sum) * previous_values_LRP[i]);
                    }   
                    
                }

            }

            inline void compute_instance_values(std::vector<std::vector<double>>& values, std::vector<std::vector<double>>& LRP_values, std::vector<double>& _instance){
                if (VERBOSE){
                    std::cout << "compute_instance_values --------------------------------------------------------------------------" << std::endl;
                }
                //Perform a forward pass on the network by computing the values
                //of the neurons in the hidden layers
                std::vector<double> current_values(_instance);
                values.push_back(current_values);
                unsigned int n_layers = coefficients->size();
                
                for (unsigned int i = 0; i < n_layers;i++){
                    // The ith element of the list holds the values of the ith layer.
                    predict_matrix_product_intercept(current_values, (*coefficients)[i], (*intercepts)[i]);
                           
                    if (VERBOSE){
                        std::cout << "layer " << i << ": ";
                        display_instance(current_values);
                    }
                    if (i != coefficients->size()-1){
                        // Apply the activation function except on the last layer
                        apply_activation(activation, current_values);
                        
                        if (VERBOSE){
                            std::cout << "layer " << i << " activation: ";
                            display_instance(current_values);
                        }
                    }
                    values.push_back(current_values);
                    
                    
                }
                // Apply out activation function on the output layer: this one depends on the problem: binary or multi class
                //std::cout << "compute_instance_values:";
                //display_instance(current_values);
                        
                apply_activation(out_activation, current_values);
                if (VERBOSE){
                    std::cout << "last layer out activation: ";
                    display_instance(current_values);
                }

                if (VERBOSE){
                    std::cout << "compute_LRP --------------------------------------------------------------------------" << std::endl;
                }
                //Perform a backward pass on the network by computing the LRP values of the neurons in the hidden layers
                std::vector<std::vector<double>> LRP_values_tmp;
                //Get the final neuron LRP value => this value have to be set to 1, whatever the prediction !
                std::vector<double> current_values_LRP;
                std::cout << "last value prediction: " << values[values.size()-1][0] << std::endl;
                current_values_LRP.push_back(values[values.size()-1][0]);
                std::vector<double> previous_values_LRP = current_values_LRP;

                LRP_values_tmp.push_back(current_values_LRP);

                if (VERBOSE){
                    std::cout << "LRP last layer:"<< std::endl;
                    std::cout << "current_values_LRP:"<< std::endl;
                    display_instance(current_values_LRP);
                }
                
                // The hidden layers 
                for (unsigned int i = n_layers-1; i > 0;i--){
                    if (VERBOSE){
                        std::cout << "LRP hidden layer:" << i << std::endl;
                    }
                    current_values_LRP.clear();
                    _compute_LRP(current_values_LRP, previous_values_LRP, (*coefficients)[i], (*intercepts)[i], values[i]);
                    previous_values_LRP = current_values_LRP;
                    LRP_values_tmp.push_back(current_values_LRP);
                    if (VERBOSE){
                        std::cout << "current_values_LRP:"<< std::endl;
                        display_instance(current_values_LRP);
                    }
                }

                // The instance (feature) LRP values
                if (VERBOSE){
                    std::cout << "LRP instance:" << std::endl;
                }
                current_values_LRP.clear();
                _compute_LRP(current_values_LRP, previous_values_LRP, (*coefficients)[0], (*intercepts)[0], _instance);
                LRP_values_tmp.push_back(current_values_LRP);
                if (VERBOSE){
                    std::cout << "current_values_LRP:"<< std::endl;
                    display_instance(current_values_LRP);
                }

                // Reverse the vector in order to have LRP values from de first layer to the last layer
                for (int i = LRP_values_tmp.size()-1; i >= 0; i--){
                    LRP_values.push_back(LRP_values_tmp[i]);
                }

            }

            inline void compute_lower_upper_neurons(
                    std::vector<std::vector<double>>& matrix_upper,
                    std::vector<std::vector<double>>& matrix_lower, 
                    std::vector<std::vector<double>>& matrix_upper_before_activation,
                    std::vector<std::vector<double>>& matrix_lower_before_activation, 
                    std::vector<double>& upper_bounds, 
                    std::vector<double>& lower_bounds){
                if (VERBOSE){
                    std::cout << "compute_lower_upper_neurons --------------------------------------------------------------------------" << std::endl;
                }
                unsigned int n_layers = coefficients->size();
                
                std::vector<double> lower = lower_bounds;
                std::vector<double> upper = upper_bounds;
                if (VERBOSE){
                    std::cout << "initial lower: ";
                    display_instance(lower);
            
                    std::cout << "initial upper: ";
                    display_instance(upper);
                }
                matrix_lower.push_back(lower);
                matrix_upper.push_back(upper);
                matrix_lower_before_activation.push_back(std::vector<double>(lower));
                matrix_upper_before_activation.push_back(std::vector<double>(upper));

                for (unsigned int i = 0; i < n_layers;i++){
                    // The ith element of the list holds the values of the ith layer.
                    std::vector<std::vector<double>>* results = matrix_product_intercept_upper_lower((*coefficients)[i], (*intercepts)[i], upper, lower);
                    lower = (*results)[0];
                    upper = (*results)[1];

                    if (VERBOSE){
                      std::cout << "layer " << i << " lower: ";  
                      display_instance(lower);
                      std::cout << "layer " << i << " upper: ";  
                      display_instance(upper);
                    }
                    matrix_lower_before_activation.push_back(std::vector<double>(lower));
                    matrix_upper_before_activation.push_back(std::vector<double>(upper));
                    
                    if (i != coefficients->size()-1){
                        // Apply the activation function except on the last layer
                        apply_activation_upper_lower(activation, lower, upper);
                        if (VERBOSE){
                            std::cout << "layer " << i << " lower activation: ";  
                            display_instance(lower);
                            std::cout << "layer " << i << " upper activation: ";  
                            display_instance(upper);
                        }
                    }

                    matrix_lower.push_back(lower);
                    matrix_upper.push_back(upper);
                }

                apply_activation_upper_lower(out_activation, lower, upper);
                matrix_lower.push_back(lower);
                matrix_upper.push_back(upper);
                if (VERBOSE){
                    std::cout << "last layer out activation lower: ";
                    display_instance(lower);
                    std::cout << "last layer out activation upper: ";
                    display_instance(upper);
                }
            }


            inline void update_lower_upper_activation(int i, ActivationType _activation, std::vector<std::vector<double>>& matrix_upper, std::vector<std::vector<double>>& matrix_lower, std::vector<std::vector<double>>& matrix_upper_before_activation, std::vector<std::vector<double>>& matrix_lower_before_activation){
                if (VERBOSE){
                    std::cout << "update_lower_upper_activation --------------------------------------------------------------------------" << std::endl;
                }
                if (_activation == RELU){
                    unsigned int n_neurons_previous_layer = matrix_upper[i+1].size();
                    //std::cout << "n_neurons_previous_layer:" << n_neurons_previous_layer << std::endl;
                    for (unsigned int j = 0; j < n_neurons_previous_layer; j++){
                        //std::cout << "neuron_previous_layer: " << j << std::endl;
                        if (VERBOSE){
                            std::cout << "Old interval:" << " [" << matrix_lower_before_activation[i+1][j] << ", " <<  matrix_upper_before_activation[i+1][j] << "]" << std::endl;
                        }
                        double previous_lower_value = matrix_lower_before_activation[i+1][j];
                        double previous_upper_value = matrix_upper_before_activation[i+1][j];
                        
                        double current_lower_value = matrix_lower[i+1][j];
                        double current_upper_value = matrix_upper[i+1][j];

                        // Pierre: new_upper = min(previous_upper_value, current_upper_value)
                        matrix_upper[i+1][j] = (previous_upper_value<current_upper_value)?previous_upper_value:current_upper_value;

                        // Pierre: new_lower = current_lower_value if current_lower_value > 0
                        //                   = previous_lower_value if current_lower_value == 0
                        if (current_lower_value > 0){
                            matrix_lower[i+1][j] = current_lower_value;
                        }else if (current_lower_value == 0){
                            matrix_lower[i+1][j] = previous_lower_value;
                        }else{
                            std::cout << "impossible case ???" << std::endl;
                            exit(0);
                        }
                        if (VERBOSE){
                            std::cout << "New interval:" << " [" << matrix_lower[i+1][j] << ", " <<  matrix_upper[i+1][j] << "]" << std::endl;
                        }
                        //std::cout << "----------------------------------" << std::endl;
                        //std::cout << "previous_lower_value:" << previous_lower_value << std::endl;
                        //std::cout << "previous_upper_value:" << previous_upper_value << std::endl;
                        //std::cout << "current_lower_value:" << current_lower_value << std::endl;
                        //std::cout << "current_upper_value:" << current_upper_value << std::endl;
                        //std::cout << "new_lower_value:" << matrix_lower[i+1][j] << std::endl;
                        //std::cout << "new_upper_value:" << matrix_upper[i+1][j] << std::endl;
                        //std::cout << "----------------------------------" << std::endl;

                    }
                    
                } else {
                    std::cout << "not implemented" << std::endl;
                    exit(0);
                }

            }

            
    };

    class ExplainerModelSpecific {
      public:
        std::vector<double>* classes;
        unsigned int n_iterations;
        unsigned int time_limit; //in seconds
        int seed;
        LinearModel linear_model;
        NeuralNetwork neural_network_model;
        ModelType model_type;
        std::default_random_engine random_engine;
        Method method;

       
        
        ExplainerModelSpecific(
                        std::vector<double>* _classes, 
                        char* _activation_obj,
                        char* _out_activation_obj,
                        std::vector<std::vector<std::vector<double>>>* _coefficients,
                        std::vector<std::vector<double>>* _intercepts) : 
                        classes(_classes), 
                        n_iterations(50), 
                        time_limit(0),
                        seed(-1),
                        linear_model(LinearModel()),
                        neural_network_model(NeuralNetwork(_activation_obj, _out_activation_obj, _coefficients, _intercepts)),
                        model_type(NEURAL_NETWORK),
                        method(METHOD1)
                        {
                            random_engine = std::default_random_engine(seed == -1 ? std::chrono::steady_clock::now().time_since_epoch().count() : 1);
                        };

        ExplainerModelSpecific(
                        std::vector<double>* _classes, 
                        std::vector<std::vector<double>>* _coefficients,
                        std::vector<double>* _intercepts) : 
                        classes(_classes), 
                        n_iterations(50), 
                        time_limit(0),
                        seed(-1),
                        linear_model(LinearModel(_coefficients, _intercepts)),
                        neural_network_model(NeuralNetwork()),
                        model_type(LINEAR_MODEL),
                        method(METHOD1)
                        {
                            random_engine = std::default_random_engine(seed == -1 ? std::chrono::steady_clock::now().time_since_epoch().count() : 1);
                        };


        inline void set_n_iterations(int _n_iterations){n_iterations = _n_iterations;}
        inline void set_time_limit(int _time_limit){time_limit = _time_limit;}
        inline void set_method(long _method){
            if (_method == 1) method = METHOD1;
            else if (_method == 2) method = METHOD2;
            else {
                std::cout << "error: the parameter method is wrong." << std::endl;
                exit(0);
            }
        }

        inline void set_LRP(char* LRP_method){
            neural_network_model.set_LRP(LRP_method);
        }

        inline LRPType get_LRP(){
            return neural_network_model.get_LRP();
        }
        
        
        inline std::vector<std::vector<bool>*>* remove_equal_reasons(std::vector<std::vector<bool>*>* reasons){
            std::vector<std::vector<bool>*>* not_equals_reasons = new std::vector<std::vector<bool>*>();
            std::vector<std::vector<bool>*>* to_delete = new std::vector<std::vector<bool>*>();
            not_equals_reasons->push_back((*reasons)[0]);
            for (unsigned int i=1; i < reasons->size(); i++){
                std::vector<bool>* reason_1 = (*reasons)[i];
                bool is_inside = false;
                for (unsigned int j=0; j < not_equals_reasons->size(); j++){
                    bool is_equal = true;
                    std::vector<bool>* reason_2 = (*not_equals_reasons)[j];
                    for (unsigned int k=0; k < reason_1->size(); k++){
                        if ((*reason_1)[k] != (*reason_2)[k]){
                            is_equal = false;
                        }
                    }
                    if (is_equal == true){
                        is_inside = true;
                    }
                }
                if (is_inside == false){
                    not_equals_reasons->push_back(reason_1);
                }else{
                    to_delete->push_back(reason_1);
                }
            }
            delete reasons;
            for (unsigned int i=1; i < to_delete->size(); i++){
                delete (*to_delete)[i];
            }
            delete to_delete;
            return not_equals_reasons;
        }






        




        inline std::vector<std::vector<bool>*>* minimal_model_specific_reasons_linear_model(std::vector<double>* instance, std::vector<double>* upper_bounds, std::vector<double>* lower_bounds){
            //std::cout << "minimal_model_specific_reason_c++" << std::endl;
            std::vector<std::vector<bool>*>* minimal_reasons = new std::vector<std::vector<bool>*>();
            std::vector<bool>* reason = NULL;
           
            unsigned int minimal_size = std::numeric_limits<int>::max(); 
            unsigned int n_true = 0;
            for (unsigned int i=0; i < n_iterations; i++){
                //std::cout << "----------------------------------------------------" << std::endl;
                reason = model_specific_reason_linear_model(*instance, *upper_bounds, *lower_bounds);
                n_true = 0;
                for (unsigned int j=0; j < reason->size(); j++){
                    if ((*reason)[j] == true){n_true += 1;}
                }
                //std::cout << "minimal_size:" << minimal_size << std::endl;
                //std::cout << "n_true:" << n_true << std::endl;


                if (n_true < minimal_size){
                    //std::cout << "A new minimal:" << n_true << std::endl;

                    // A new minimal:
                    for (std::vector<bool>* minimal_reason: *minimal_reasons){
                        free(minimal_reason);
                    }
                    minimal_reasons->clear();
                    minimal_reasons->push_back(reason);
                    minimal_size = n_true;
                }else if(minimal_size == n_true){
                    //std::cout << "Equal:" << n_true << std::endl;
                    minimal_reasons->push_back(reason);
                }else{
                    free(reason);
                }
            }

            minimal_reasons = remove_equal_reasons(minimal_reasons);

            //std::cout << "Number of reasons:" << minimal_reasons->size() << std::endl;
            //std::cout << "reason size:" << (*minimal_reasons)[0]->size() << std::endl;

            return minimal_reasons;
        }

        inline double predict(std::vector<double>& instance){
            double instance_class = model_type==LINEAR_MODEL ? linear_model.predict(instance, *classes): neural_network_model.predict(instance, *classes);
            return instance_class;
        }


        inline void display_matrix(std::vector<std::vector<double>>& matrix){
            for (unsigned int i = 0; i< matrix.size(); i++){
                std::vector<double>& line = matrix[i];
                for (unsigned int j = 0; j< line.size(); j++){
                    std::cout << line[j] << " ";
                }
                std::cout << std::endl;    
            }
            std::cout << std::endl;
        }

        inline void display_line(std::vector<double>& line){
            for (unsigned int i = 0; i< line.size(); i++){
                std::cout << std::fixed << line[i] << " ";    
            }
            std::cout << std::endl;
        }

        inline std::string sign_to_string(InequationSign inequation_sign){
            if (inequation_sign == LT){
                return "<";
            }else if (inequation_sign == LE){
                return "<=";
            }else if (inequation_sign == GT){
                return ">";
            }else if (inequation_sign == GE){
                return ">=";
            }
            return "error";
        }

        inline InequationSign invert_sign(InequationSign inequation_sign){
            if (inequation_sign == LT){
                return GE;
            }else if (inequation_sign == LE){
                return GT;
            }else if (inequation_sign == GT){
                return LE;
            }else if (inequation_sign == GE){
                return LT;
            }
            std::cout << "error in invert_sign" << std::endl;
            exit(0);
        }

        inline double round_to(double value, int precision)
        {
            std::stringstream str_representation; 
            str_representation << std::setprecision(precision) << value;
            std::cout << "round:" << str_representation.str() << std::endl;
            return stod(str_representation.str());
        }

        inline void solve_inequality(unsigned int id_inequality_order, 
                                        unsigned int id_neuron_order, 
                                        std::vector<int>& order, 
                                        std::vector<int>& order_neurons,
                                        std::vector<double>& current_values,
                                        std::vector<std::vector<double>>& current_coefficients, 
                                        std::vector<double>& current_intercepts, 
                                        std::vector<double>& current_upper, 
                                        std::vector<double>& current_lower, 
                                        InequationSign inequation_sign, double inequation_value){
                                            

            //std::cout << "id_inequality_order: " << id_inequality_order << std::endl;
            //std::cout << "id_neuron: " << id_neuron << std::endl;
            
            //std::cout << "inequation_sign: " << sign_to_string(inequation_sign) << std::endl;
            //std::cout << "inequation_value: " << inequation_value << std::endl;
            //std::cout << "current values: ";
            //display_line(current_values);
            //std::cout << "current_lower: ";
            //display_line(current_lower);
            //std::cout << "current_upper: ";
            //display_line(current_upper);
            //if (current_lower[id_inequality] == current_upper[id_inequality]){
                // We do not need to solve this inequality :) 
            //    return;
            //}
            
            if (VERBOSE){
                std::cout << "solve_inequality --------------------------------------------------------------------------" << std::endl;
                std::cout << "id_inequality_order: " << id_inequality_order << std::endl;
                std::cout << "id_neuron_order: " << id_inequality_order << std::endl;
            }
            int id_inequality = order[id_inequality_order];
            int id_neuron = order_neurons[id_neuron_order];
            if (VERBOSE){
                std::cout << "id_inequality: " << id_inequality << std::endl;
                std::cout << "id_neuron: " << id_neuron << std::endl;
            }

            std::stringstream str_representation; 
            double current_value_of_instance = current_values[id_inequality];

            //str_representation << "inequation:" << sign_to_string(inequation_sign) << " " << inequation_value << std::endl;
            
            unsigned int n_coefficients = current_coefficients.size();
            double coefficient_to_divide = 0;
            double sum = 0;
            //for (unsigned int i = 0; i < n_coefficients; i++){
            for (unsigned int i_order = 0; i_order < n_coefficients; i_order++){
                int i = order[i_order];
                //std::cout << current_coefficients[i][id_neuron] << " ";
                double coefficient = current_coefficients[i][id_neuron];
                if (i_order < id_inequality_order){
                    // Take the upper or the lower of the previous computed interval according to the sign of the weight and the sign of the inequality
                    if (coefficient >= 0){
                        // Positive coefficient 
                        if (inequation_sign == GE || inequation_sign == GT){
                            // Positive sign: take the lower 
                            sum += coefficient*current_lower[i];      
                            if (VERBOSE){
                                str_representation << "(" << coefficient << " * " << current_lower[i] << ")";
                            }
                        }else{
                            // Negative sign: take the upper;
                            sum += coefficient*current_upper[i];
                            if (VERBOSE){
                                str_representation << "(" << coefficient << " * " << current_upper[i] << ")";      
                            }
                        }
                    }else{
                        // Negative coefficient
                        if (inequation_sign == GE || inequation_sign == GT){
                            // Positive sign: take the upper 
                            sum += coefficient*current_upper[i];
                            if (VERBOSE){
                                str_representation << "(" << coefficient << " * " << current_upper[i] << ")";
                            }
                        }else{
                            // Negative sign: take the lower;
                            sum += coefficient*current_lower[i];
                            if (VERBOSE){
                                str_representation << "(" << coefficient << " * " << current_lower[i] << ")";
                            }
                        }
                    }

                }else if (i_order == id_inequality_order){
                    // The variable to compute    
                    coefficient_to_divide = coefficient;
                    if (VERBOSE){
                        str_representation << "(" << coefficient << " * " << "X" << ")";
                    }
                }else{
                    sum += (coefficient*current_values[i]);
                    if (VERBOSE){
                        str_representation << "(" << coefficient << " * " << current_values[i] << ")";
                    }
                    // Take the value of the instance
                }
                if (VERBOSE && i < n_coefficients-1){
                    str_representation << "+";
                }    
                
            }
            if (VERBOSE){
                str_representation << ((current_intercepts[id_neuron]>0)?"+":"") << current_intercepts[id_neuron];
                str_representation << " " << sign_to_string(inequation_sign) << " " << inequation_value;
                std::cout << str_representation.str() << std::endl;
            }
            sum += current_intercepts[id_neuron];
            inequation_value = inequation_value-sum;

            double value = inequation_value/coefficient_to_divide;
            if (coefficient_to_divide < 0){
                // We invert the sign of the inequation when we divide by a negative number :)
                inequation_sign = invert_sign(inequation_sign);
            }
            if (VERBOSE){
                std::cout << "X " << sign_to_string(inequation_sign) << " " << value << std::endl;
                std::cout << "Old interval:" << std::setprecision(30) << " [" << current_lower[id_inequality] << ", " <<  current_upper[id_inequality] << "]" << std::endl;
            }
            //Change the lower or upper bound for the next inequation: two main cases
            if (inequation_sign == LE || inequation_sign == LT){
                // X < value 
                if (value <= current_lower[id_inequality]){
                    // In this case, we exceed the interval: lower = upper (it is possible due to some little error with the precision of doubles)
                    if (current_value_of_instance == current_lower[id_inequality]){
                        current_upper[id_inequality] = current_lower[id_inequality]; 
                    }else{
                        current_upper[id_inequality] = current_value_of_instance;
                    }
                }else if (value >= current_upper[id_inequality]){
                    // We know alread that (if value >= upper, then the interval has already the smaller), in this case we have nothing to do
                }else{
                    // We are strictly inside ]lower;upper[
                    if (current_value_of_instance > value){
                        current_upper[id_inequality] = current_value_of_instance;
                    }else{
                        current_upper[id_inequality] = value;
                    }
                }
            }else if (inequation_sign == GE || inequation_sign == GT){
                // X > value  
                if (value >= current_upper[id_inequality]){
                    // In this case, we close the interval: lower = upper (it is possible due to some little error with the precision of doubles)
                    if (current_value_of_instance == current_upper[id_inequality]){
                        current_lower[id_inequality] = current_upper[id_inequality]; 
                    }else{
                        current_lower[id_inequality] = current_value_of_instance;
                    }
                }else if (value <= current_lower[id_inequality]){
                    // We know alread that (if value >= lower, then the interval has already the smaller), in this case we have nothing to do
                }else{
                    // We are strictly inside ]lower;upper[
                    if (current_value_of_instance < value){
                        current_lower[id_inequality] = current_value_of_instance;
                    }else{
                        current_lower[id_inequality] = value;
                    
                    }
                }
            }
            if (VERBOSE){
                std::cout << "New interval:" << " [" << current_lower[id_inequality] << ", " <<  current_upper[id_inequality] << "]" << std::endl;
            }
        }

        inline void order_weights_max(std::vector<int>& order, std::vector<double>& weights){
            std::vector<bool> weights_done(weights.size(), false);
            unsigned int index_max = 0;
            double max = 0;
            for (unsigned int i = 0; i < weights.size(); i++){
                max = -std::numeric_limits<double>::infinity();
                for (unsigned int j = 0; j < weights.size(); j++){
                    if (weights_done[j] == false && weights[j] > max) {
                        max = weights[j];
                        index_max = j;
                    }
                }
                weights_done[index_max] = true;
                //std::cout << "index_max: " << index_max << std::endl;
                order.push_back(index_max);

            }       
        }

        inline void order_weights_min(std::vector<int>& order, std::vector<double>& weights){
            std::vector<bool> weights_done(weights.size(), false);
            unsigned int index_min = 0;
            double min = 0;
            for (unsigned int i = 0; i < weights.size(); i++){
                min = std::numeric_limits<double>::infinity();
                for (unsigned int j = 0; j < weights.size(); j++){
                    if (weights_done[j] == false && weights[j] < min) {
                        min = weights[j];
                        index_min = j;
                    }
                }
                weights_done[index_min] = true;
                //std::cout << "index_max: " << index_max << std::endl;
                order.push_back(index_min);

            }       
        }
        
        inline void _model_specific_reason_neural_network(
                double instance_class,
                std::vector<std::vector<double>>& LRP_values,
                std::vector<std::vector<double>>& matrix_upper,
                std::vector<std::vector<double>>& matrix_lower,
                std::vector<std::vector<double>>& matrix_upper_before_activation,
                std::vector<std::vector<double>>& matrix_lower_before_activation,
                std::vector<std::vector<double>>& instance_values,
                std::vector<double>& result_upper_bounds, 
                std::vector<double>& result_lower_bounds){
            int n_layers = neural_network_model.coefficients->size();
            std::vector<int> order;
            std::vector<int> order_neurons;
            
            // For each layer from the last to the begin
            for (int i = n_layers-1; i >= 0;i--){            
                std::vector<std::vector<double>>& current_coefficients = (*neural_network_model.coefficients)[i];
                std::vector<double>& current_intercepts = (*neural_network_model.intercepts)[i];
                std::vector<double>& current_upper = matrix_upper[i];
                std::vector<double>& current_lower = matrix_lower[i];
                std::vector<double>& current_upper_before_activation = matrix_upper_before_activation[i];
                std::vector<double>& current_lower_before_activation = matrix_lower_before_activation[i];
                
                std::vector<double>& current_values = instance_values[i];
                std::vector<double>& current_LRP = LRP_values[i];
                
                if (VERBOSE){
                    std::cout << "Compute layer " << i << " --------------------------------------------------------------------------" << std::endl;
                    std::cout << "coefficients: " << std::endl;
                    display_matrix(current_coefficients);
                    std::cout << "intercepts: ";
                    display_line(current_intercepts);
                    std::cout << "lower: ";
                    display_line(current_lower);
                    std::cout << "upper: ";
                    display_line(current_upper);
                    
                    std::cout << "lower before activication: ";
                    display_line(current_lower_before_activation);
                    std::cout << "upper before activication: ";
                    display_line(current_upper_before_activation);


                    std::cout << "current_LRP: " << LRP_values.size() << std::endl;
                    display_line(current_LRP);
                    
                }

                //We have to apply the activation before (except for the last layer)
                if (i != n_layers-1){
                    // Update matrix_upper and matrix_lower according to the activation function
                    neural_network_model.update_lower_upper_activation(i, neural_network_model.activation, matrix_upper, matrix_lower, matrix_upper_before_activation, matrix_lower_before_activation);
                }

                // For each neuron
                unsigned int n_neurons = current_coefficients[0].size();
                unsigned int n_coefficients = current_coefficients.size();
                
                order_neurons.clear();
                for (unsigned int l = 0; l < n_neurons;l++){order_neurons.push_back(l);}
                std::shuffle(std::begin(order_neurons), std::end(order_neurons), random_engine);


                order.clear();
                if (get_LRP() == NORMAL_UP){
                    order_weights_max(order, current_LRP);
                    //std::cout << "Order  --------------------------------------------------------------------------" << std::endl;
                    //for (unsigned int l = 0; l < n_coefficients;l++){std::cout << " " << order[l];}
                    //std::cout << std::endl;
                }else if (get_LRP() == NORMAL_DOWN){
                    order_weights_min(order, current_LRP);
                }else{
                    for (unsigned int l = 0; l < n_coefficients;l++){order.push_back(l);}
                    std::shuffle(std::begin(order), std::end(order), random_engine);
                }

                if (method == METHOD1){
                    for (unsigned int j = 0; j < n_neurons; j++){
                        if (i == n_layers-1){
                            //solve_inequalities(j, current_values, current_coefficients, current_intercepts, current_upper, current_lower, inequation_sign, inequation_value);
                            for (unsigned int k = 0; k < n_coefficients; k++){
                                solve_inequality(k, j, order, order_neurons, current_values, current_coefficients, current_intercepts, current_upper, current_lower, (instance_class == 0)?LT:GE, 0);
                            }
                        }else{
                            // Layer: two inequation according to lower and upper bound 
                            for (unsigned int k = 0; k < n_coefficients; k++){
                                solve_inequality(k, j, order, order_neurons, current_values, current_coefficients, current_intercepts, current_upper, current_lower, GE, matrix_lower[i+1][j]);
                                solve_inequality(k, j, order, order_neurons, current_values, current_coefficients, current_intercepts, current_upper, current_lower, LE, matrix_upper[i+1][j]);
                            }
                        }
                    }
                }else if (method == METHOD2){
                    for (unsigned int k = 0; k < n_coefficients; k++){
                        //if (current_lower[order[k]] == current_upper[order[k]]){
                            // We do not need to solve this inequality :) 
                        //    continue;
                        //}
                        for (unsigned int j = 0; j < n_neurons; j++){
                            if (i == n_layers-1){
                                solve_inequality(k, j, order, order_neurons, current_values, current_coefficients, current_intercepts, current_upper, current_lower, (instance_class == 0)?LT:GE, 0);
                            }else{
                                //if (matrix_lower[i+1][j] != matrix_upper[i+1][j]){
                                    solve_inequality(k, j, order, order_neurons, current_values, current_coefficients, current_intercepts, current_upper, current_lower, GE, matrix_lower[i+1][j]);
                                    solve_inequality(k, j, order, order_neurons, current_values, current_coefficients, current_intercepts, current_upper, current_lower, LE, matrix_upper[i+1][j]);          
                                //}
                            }
                        }
                    }
                }
            

                
            }
            result_upper_bounds = matrix_upper[0];
            result_lower_bounds = matrix_lower[0];
        }

        inline void model_specific_reason_neural_network(std::vector<double>& instance, std::vector<double>& upper_bounds, std::vector<double>& lower_bounds, std::vector<double>& result_upper_bounds, std::vector<double>& result_lower_bounds){
            double instance_class = neural_network_model.predict(instance, *classes);
            if (VERBOSE){
                std::cout << "model_specific_reason_neural_network_c++ --------------------------------------------------------------------------" << std::endl;
                std::cout << "instance: ";
                display_line(instance);
                std::cout << "instance_class: " << instance_class << std::endl;
            }
            // Compute the upper bound and lower bound matrixes 
            std::vector<std::vector<double>> matrix_upper;
            std::vector<std::vector<double>> matrix_lower;
            std::vector<std::vector<double>> matrix_upper_before_activation;
            std::vector<std::vector<double>> matrix_lower_before_activation;
            std::vector<std::vector<double>> instance_values;
            std::vector<std::vector<double>> LRP_values;
            
            
            neural_network_model.compute_instance_values(instance_values, LRP_values, instance);
            neural_network_model.compute_lower_upper_neurons(matrix_upper, matrix_lower, matrix_upper_before_activation, matrix_lower_before_activation, upper_bounds, lower_bounds);
            
            

            _model_specific_reason_neural_network(instance_class, LRP_values, matrix_upper, matrix_lower, matrix_upper_before_activation, matrix_lower_before_activation, instance_values, result_upper_bounds, result_lower_bounds);    
        }    

        inline void LRP_reason_neural_network(std::vector<double>& instance, std::vector<double>& reason){
            double instance_class = neural_network_model.predict(instance, *classes);
            if (VERBOSE){
                std::cout << "LRP_reason_neural_network_c++ --------------------------------------------------------------------------" << std::endl;
                std::cout << "instance: ";
                display_line(instance);
                std::cout << "instance_class: " << instance_class << std::endl;
            }
            // Compute the upper bound and lower bound matrixes 
            std::vector<std::vector<double>> instance_values;
            std::vector<std::vector<double>> LRP_values;
            
            neural_network_model.compute_instance_values(instance_values, LRP_values, instance);
            // We have reverse the LRP values, so the good values are the first ones
            reason = LRP_values[0];
            if (VERBOSE){
                std::cout << "LRP_reason:" << std::endl;
                display_line(reason);
            }
        }    

        inline void minimal_model_specific_reasons_neural_network(std::vector<double>& instance, std::vector<double>& upper_bounds, std::vector<double>& lower_bounds, std::vector<double>& result_upper_bounds, std::vector<double>& result_lower_bounds){
            double instance_class = neural_network_model.predict(instance, *classes);
            std::cout << "model_specific_reason_neural_network_c++ --------------------------------------------------------------------------" << std::endl;
            std::cout << "Method:" << method << std::endl;

            if (VERBOSE){
                std::cout << "model_specific_reason_neural_network_c++ --------------------------------------------------------------------------" << std::endl;
                std::cout << "instance: ";
                display_line(instance);
                std::cout << "instance_class: " << instance_class << std::endl;
            }
            // Compute the upper bound and lower bound matrixes 
            std::vector<std::vector<double>> matrix_upper_base;
            std::vector<std::vector<double>> matrix_lower_base;
            std::vector<std::vector<double>> matrix_upper_before_activation_base;
            std::vector<std::vector<double>> matrix_lower_before_activation_base;
            std::vector<std::vector<double>> instance_values;
            std::vector<std::vector<double>> LRP_values;

            neural_network_model.compute_instance_values(instance_values, LRP_values, instance);
            neural_network_model.compute_lower_upper_neurons(matrix_upper_base, matrix_lower_base, matrix_upper_before_activation_base, matrix_lower_before_activation_base, upper_bounds, lower_bounds);
            
            std::vector<std::vector<double>> matrix_upper;
            std::vector<std::vector<double>> matrix_lower;
            std::vector<std::vector<double>> matrix_upper_before_activation;
            std::vector<std::vector<double>> matrix_lower_before_activation;

            std::vector<double> tmp_result_upper_bounds;
            std::vector<double> tmp_result_lower_bounds;
            
            std::cout << "n_iterations: " << n_iterations << std::endl;
            
            double initial_interval_length = 0;
            double result_interval_length = 0;
            
            double best_reduction_percentage = 101;
            double best_n_opened = 0;
            
            double reduction_percentage_avg = 0;
            double reduction_percentage_sum = 0;

            for (unsigned int i = 0; i < n_iterations;i++){
                reduction_percentage_avg = 0;
                reduction_percentage_sum = 0;
                matrix_upper = matrix_upper_base;
                matrix_lower = matrix_lower_base;
                matrix_upper_before_activation = matrix_upper_before_activation_base;
                matrix_lower_before_activation = matrix_lower_before_activation_base;
                tmp_result_upper_bounds.clear();
                tmp_result_lower_bounds.clear();
                // Compute 
                _model_specific_reason_neural_network(instance_class, LRP_values, matrix_upper, matrix_lower, matrix_upper_before_activation, matrix_lower_before_activation, instance_values, tmp_result_upper_bounds, tmp_result_lower_bounds);    
                int n_opened = 0;
                for (unsigned int j = 0; j < instance.size();j++){
                    initial_interval_length = upper_bounds[j] - lower_bounds[j];
                    result_interval_length = tmp_result_upper_bounds[j] - tmp_result_lower_bounds[j];
                    //std::cout << "initial_interval_length:" << initial_interval_length << std::endl;
                    //std::cout << "rresult_interval_length:" << result_interval_length << std::endl;    
                    if (result_interval_length >= 1){
                        n_opened++;
                    } 
                    if (initial_interval_length != 0){
                        reduction_percentage_sum += 100-((result_interval_length*100) / initial_interval_length);
                    }
                }
                //std::cout << "reduction_percentage_sum:" << reduction_percentage_sum << std::endl;
                reduction_percentage_avg = reduction_percentage_sum/instance.size();
                //std::cout << "reduction_percentage_avg:" << reduction_percentage_avg << std::endl;
                if (n_opened > best_n_opened){
                    result_upper_bounds = tmp_result_upper_bounds;
                    result_lower_bounds = tmp_result_lower_bounds;
                    best_n_opened = n_opened;
                    std::cout << std::setprecision(10) << "new best_n_opened:" << best_n_opened << std::endl;
                    
                }
                if (reduction_percentage_avg < best_reduction_percentage){
                //    result_upper_bounds = tmp_result_upper_bounds;
                //    result_lower_bounds = tmp_result_lower_bounds;
                    best_reduction_percentage = reduction_percentage_avg;
                    std::cout << std::setprecision(10) << "new best_reduction_percentage:" << best_reduction_percentage << std::endl;
                }

            }
            std::cout << std::setprecision(10) << "new best_n_opened:" << best_n_opened << std::endl;
            std::cout << "final best_reduction_percentage:" << best_reduction_percentage << std::endl;
            

        }
        


        inline std::vector<bool>* model_specific_reason_linear_model(std::vector<double>& instance, std::vector<double>& upper_bounds, std::vector<double>& lower_bounds){
            /*std::cout << "model_specific_reason_c++" << std::endl;*/
            
            
            std::vector<bool> reason(instance.size(), false); // Initially, nothing in the reason 

            unsigned int n_models = classes->size() == 2? 1: classes->size();

            double instance_class = linear_model.predict(instance, *classes);

            //std::vector<int> order = {0, 1, 2};
            std::vector<int> order;
            for (unsigned int i = 0; i < instance.size();i++){order.push_back(i);}
            std::shuffle(std::begin(order), std::end(order), random_engine);
            /*for (unsigned int i = 0; i < order.size();i++){
               std::cout << "order: " << order[i] << std::endl;
            }*/
            //Change the order of each element
            std::vector<double> instance_order;
            std::vector<double> upper_bounds_order;
            std::vector<double> lower_bounds_order;
            std::vector<std::vector<double>> coefficients_order;
            for (unsigned int i = 0; i < order.size();i++){
                instance_order.push_back(instance[order[i]]);
                upper_bounds_order.push_back(upper_bounds[order[i]]);
                lower_bounds_order.push_back(lower_bounds[order[i]]);
            }
            for (unsigned int j = 0; j < n_models;j++){
                coefficients_order.push_back(std::vector<double>());
                for (unsigned int i = 0; i < order.size();i++){
                    coefficients_order[j].push_back((*linear_model.coefficients)[j][order[i]]);
                }
            }

            std::vector<double> current_predictions(n_models, 0);
            

            for (unsigned int i = 0; i < instance_order.size();i++){
                //std::cout << "instance_order:" << instance_order[i] << std::endl;
                // Main loop: condider the n attributes in sequence
                
                for (unsigned int j = 0; j < n_models;j++){
                    // We have to calculate the worst/best instance for each model and the associated prediction
                    current_predictions[j] = linear_model._worst_best_instance_predict(instance_order, upper_bounds_order, lower_bounds_order, coefficients_order[j], (*linear_model.intercepts)[j], i, reason);
                    //std::cout << "current_predictions[j]:" << current_predictions[j] << std::endl;
                }
                int current_class = linear_model.predict_class(instance_order, current_predictions, *classes);
                //std::cout << "current_class:" << current_class << std::endl;
                if (current_class != instance_class){
                    //std::cout << "add " << i << std::endl;
                    reason[i] = true;
                }
            }
            std::vector<bool> reason_order;
            for (unsigned int i = 0; i < order.size();i++){
                //std::cout << "reason: " << reason[order[i]] << std::endl;
                reason_order.push_back(reason[order[i]]);
            }
            return new std::vector<bool>(reason_order);
        }

        inline bool check_abductive_model_specific_neural_network(std::vector<double>& instance, std::vector<double>& upper_bounds, std::vector<double>& lower_bounds, std::vector<double>& reason_upper_bounds, std::vector<double>& reason_lower_bounds, long n_samples){
            double instance_class = neural_network_model.predict(instance, *classes);
            //std::cout << "instance_class: " << instance_class << std::endl;
            //std::cout << "n_samples: " << n_samples << std::endl;
            //std::cout << "upper_bounds: " << std::endl;
            //display_line(upper_bounds);
            //std::cout << "lower_bounds: " << std::endl;
            //display_line(lower_bounds);
            //std::cout << "reason_upper_bounds: " << std::endl;
            //display_line(reason_upper_bounds);
            //std::cout << "reason_lower_bounds: " << std::endl;
            //display_line(reason_lower_bounds);
            
            std::vector<double> instance_to_test;
            std::vector<std::uniform_real_distribution<>> random_generators;
            std::random_device rd;  // Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
            
            for (unsigned int j = 0; j < instance.size();j++){random_generators.push_back(std::uniform_real_distribution<>(reason_lower_bounds[j], reason_upper_bounds[j]));}

            for (long i=0; i < n_samples; i++){
                //std::cout << "----------------------------------------------------" << std::endl;
                //std::cout << "check abductive: " << i << std::endl;
                //Generate an instance 
                instance_to_test.clear();
                for (unsigned int j = 0; j < instance.size();j++){instance_to_test.push_back(random_generators[j](gen));}
                //std::cout << "instance_to_test: " << std::endl;
                //display_line(instance_to_test);
                //Test the instance
                if (neural_network_model.predict(instance_to_test, *classes) != instance_class){
                    std::cout << "instance_to_test: " << std::endl;
                    display_line(instance_to_test);
                    
                    return false;
                }
            }
            return true;
        }

    };
}


#endif //CPP_CODE_EXPLAINER_LINEAR_H
