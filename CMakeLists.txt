cmake_minimum_required(VERSION 3.10.0)
project(neural_net VERSION 0.1.0 LANGUAGES CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nn)

add_executable(neural_net 
main.cpp 
src/nn/layers/DenseLayer.cc
src/nn/activation/ReLU.cc
src/nn/activation/Softmax.cc
src/nn/loss/LossFunction.cc
)