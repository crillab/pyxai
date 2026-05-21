//
// Created by audemard on 17/03/23.
//

#ifndef CMAKE_EXAMPLE_CONSTANTS_H
#define CMAKE_EXAMPLE_CONSTANTS_H
#include <string>
#include <cstring>
#include <iostream>

namespace pyxai {
    typedef enum tdtype {Classifier_BT, Classifier_RF, Regression_BT} Type;
    typedef enum {NONE_ACTIVATION, IDENTITY, SOFTMAX, LOGISTIC, HEAVISIDE, RELU} ActivationType;
    typedef enum {NONE_FUNCTION, TRANSPOSE, TORCH_PERMUTE, FLATTEN, TORCH_FLATTEN, TORCH_UNSQUEEZE, SELECT_FIRST_BATCH} FonctionName;
    typedef enum {NONE_PROBLEM, CLASSIFICATION, REGRESSION} ProblemType;

    static ActivationType str_to_activation_type(const char* str_activation){
        if (strcmp(str_activation, "softmax") == 0) return ActivationType::SOFTMAX;
        else if (strcmp(str_activation, "logistic") == 0) return ActivationType::LOGISTIC;
        else if (strcmp(str_activation, "identity") == 0) return ActivationType::IDENTITY;
        else if (strcmp(str_activation, "relu") == 0) return ActivationType::RELU;
        else if (strcmp(str_activation, "heaviside") == 0) return ActivationType::HEAVISIDE;
        else if (strcmp(str_activation, "none") == 0) return ActivationType::NONE_ACTIVATION;
        else {
            std::cout << "str_to_activation_type:" << str_activation << " error." << std::endl;
            exit(0);
        }
        return NONE_ACTIVATION;
    }

    static FonctionName str_to_fonction_name(const char* str_fonction){
        if (strcmp(str_fonction, "transpose") == 0) return FonctionName::TRANSPOSE;
        else if (strcmp(str_fonction, "torch_permute") == 0) return FonctionName::TORCH_PERMUTE;
        else if (strcmp(str_fonction, "flatten") == 0) return FonctionName::FLATTEN;
        else if (strcmp(str_fonction, "torch_flatten") == 0) return FonctionName::TORCH_FLATTEN;
        else if (strcmp(str_fonction, "torch_unsqueeze") == 0) return FonctionName::TORCH_UNSQUEEZE;
        else if (strcmp(str_fonction, "select_first_batch") == 0) return FonctionName::SELECT_FIRST_BATCH;
        else {
            std::cout << "str_to_fonction_name:" << str_fonction << " error." << std::endl;
            exit(0);
        }
        return NONE_FUNCTION;
    }

    static ProblemType str_to_problem_type(const char* str_problem_type){
        if (strcmp(str_problem_type, "classification") == 0) return ProblemType::CLASSIFICATION;
        else if (strcmp(str_problem_type, "regression") == 0) return ProblemType::REGRESSION;
        else {
            std::cout << "str_to_problem_type:" << str_problem_type << " error." << std::endl;
            exit(0);
        }
        return NONE_PROBLEM;
    }
}
#endif //CMAKE_EXAMPLE_CONSTANTS_H
