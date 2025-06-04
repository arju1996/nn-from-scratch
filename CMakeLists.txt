cmake_minimum_required(VERSION 3.10.0)
project(object_pool VERSION 0.1.0 LANGUAGES CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)

add_executable(object_pool main.cpp)