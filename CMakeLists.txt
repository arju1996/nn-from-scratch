cmake_minimum_required(VERSION 3.10.0)
project(neural_net VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/matrix)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nn)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nn/mix)


add_executable(neural_net 
# main.cpp
main_final.cpp
src/nn/layers/DenseLayer.cc
src/nn/activation/ReLU.cc
src/nn/activation/Softmax.cc
src/nn/loss/LossFunction.cc
src/matrix/Mat.cc
src/nn/mix/A_Softmax_L_CatergoricalCrossEntropy.cc

src/nn/optimizer/OptimizerSGD.cc
src/nn/optimizer/OptimizerAdagrad.cc
src/nn/optimizer/OptimizerRMSprop.cc
src/nn/optimizer/OptimizerADAM.cc
src/nn/layers/DropoutLayer.cc


)

# add_executable(
#     training
#     blackhole/main_test_gradient.cpp

#     src/nn/activation/ReLU.cc
# )


# add_executable(
#     matrix 
#     src/matrix/Mat.h
#     src/matrix/Mat.cc
#     src/matrix/test.cc 
# )