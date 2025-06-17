cmake_minimum_required(VERSION 3.10.0)
project(neural_net VERSION 0.1.0 LANGUAGES CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nn)

add_executable(neural_net 
main.cpp 
nn/DenseLayer.cc
nn/ReLU.cc
nn/Softmax.cc
nn/LossFunction.cc
)