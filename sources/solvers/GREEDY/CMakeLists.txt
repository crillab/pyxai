cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)


include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE Greedy_SOURCES
        ${PROJECT_SOURCE_DIR}/*.cc
        ${PROJECT_SOURCE_DIR}/bcp*.cc)

file(GLOB_RECURSE Greedy_HEADERS
        ${PROJECT_SOURCE_DIR}/*.h
        ${PROJECT_SOURCE_DIR}/bcp*.h)

file(GLOB_RECURSE Greedy_HEADERS "./*.h")

include_directories(
        ${Greedy_HEADERS}
)
#add_library(MyExample STATIC ${Greedy_SOURCES})

add_executable(MyExample ${Greedy_SOURCES})
# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.

#target_compile_definitions(majoritary_bt PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})