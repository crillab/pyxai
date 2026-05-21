cmake_minimum_required(VERSION 3.1)

# set the project name and version
project(Balls VERSION 0.1)

add_custom_target(clean-cmake-files
   COMMAND ${CMAKE_COMMAND} -P clean-all.cmake
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_BUILD_TYPE Release)

cmake_minimum_required(VERSION 3.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# source/include files
file(GLOB_RECURSE SOURCES ./src/**.cpp)
file(GLOB_RECURSE INCLUDES ./src/**.hpp)

set(SRCS ${SOURCES})
set(HEADERS ${INCLUDES})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rdParty/glucose-3.0/)

if (NOT DEFINED BUILD_MODE)
   set(BUILD_MODE 1)
endif()


# flags
if(${BUILD_MODE} EQUAL 1)
  # create an executable
  add_executable(encore_debug ${SRCS} ${HEADERS}
          src/methods/Checker.cpp
          src/methods/Checker.hpp)

  # PROFILE
  message("Debug mode activated")
  set(CMAKE_BUILD_TYPE Debug)
  target_compile_options(encore_debug PUBLIC -std=c++11 -Wall -lz -pg -lgmpxx -lgmp -O2 -Wno-class-memaccess -Wno-unused-function -Wno-class-memaccess -Wno-format -Wno-misleading-indentation -Wno-literal-suffix -Wno-sign-compare -Wno-parentheses)
  target_link_libraries (encore_debug -lz -pg -O2  ${CMAKE_SOURCE_DIR}/3rdParty/glucose-3.0/core/lib_debug.a)

  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/encore_debug
    DESTINATION bin
    RENAME ${CMAKE_PROJECT_NAME}-encore_debug)  
elseif(${BUILD_MODE} EQUAL 2)
  message("Static mode activated")

  add_executable(encore_static ${SRCS} ${HEADERS})

  target_compile_options(encore_static PUBLIC -std=c++11 -Wall -lz -O3 -Wno-class-memaccess -D NDEBUG -static -Wno-unused-function -Wno-class-memaccess -Wno-format -Wno-misleading-indentation -Wno-literal-suffix -Wno-sign-compare -Wno-parentheses)
  target_link_libraries (encore_static -lz -O3  ${CMAKE_SOURCE_DIR}/3rdParty/glucose-3.0/core/lib_release.a)

  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/encore_static
    DESTINATION bin
    RENAME ${CMAKE_PROJECT_NAME}-encore_static)
else()  
  # create an executable
  add_executable(encore ${SRCS} ${HEADERS})

  message("Classical mode activated")
  target_compile_options(encore PUBLIC -std=c++11 -Wall -lz -O3 -Wno-class-memaccess -Wno-unused-function -Wno-class-memaccess -Wno-format -Wno-misleading-indentation -Wno-literal-suffix -Wno-sign-compare -Wno-parentheses)
  target_link_libraries (encore -lz -O3 ${CMAKE_SOURCE_DIR}/3rdParty/glucose-3.0/core/lib_standard.a)

  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/encore
    DESTINATION bin
    RENAME ${CMAKE_PROJECT_NAME}-encore)
endif()


file(GLOB cmake_generated ${CMAKE_SOURCE_DIR}/src/*~)

foreach(file ${cmake_generated})

  if (EXISTS ${file})
     file(REMOVE_RECURSE ${file})
  endif()

endforeach(file)
