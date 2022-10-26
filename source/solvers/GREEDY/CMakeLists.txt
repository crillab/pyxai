cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)

execute_process(COMMAND python3 -m pybind11 --includes OUTPUT_VARIABLE pybind11_DIR)

message("pybind dir:" ${pybind11_DIR})

message("include pybind:" ${FOO})

find_package(pybind11 REQUIRED)



include_directories(${FOO})
find_package(pybind11 REQUIRED)

set(src/MajoritaryExplainerBT.cc src/majoritary_bt_wrapper.c)

#include_directories()

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.

#target_compile_definitions(majoritary_bt PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})