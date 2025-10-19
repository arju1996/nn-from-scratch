#pragma once

#include <iostream>
#include <array>
#include <vector>

#include <cassert>

#include "random.h"
#include "Mat.h"

class DenseLayer {
    int input_size_;
    int no_of_nodes_;
    /*
    std::vector<std::vector<double>> weights_;
    std::vector<double> bias_;
    */
    mynn::Mat weights_;
    mynn::Mat bias_;
    double l1_weight_regularizer_, l1_bias_regularizer_;
    double l2_weight_regularizer_, l2_bias_regularizer_;
public:
    DenseLayer(int input_size, int no_of_nodes);
    DenseLayer(int input_size, int no_of_nodes,double l1_weight_regularizer,double l1_bias_regularizer, double l2_weight_regularizer, double l2_bias_regularizer);
    void PrintDenseLayer();
    void whatever();
    inline double VectorDotVector(std::vector<double> a, std::vector<double> b); //
    std::vector<double> run(std::vector<double> input); //
    mynn::Mat forward(mynn::Mat input);
    std::tuple<mynn::Mat, mynn::Mat, mynn::Mat> backward(mynn::Mat dl_dz, mynn::Mat input);
    void backwardInputBatch(std::vector<std::vector<double>> dl_dz); //
    std::vector<std::vector<double>> runInputBatch(std::vector<std::vector<double>> inputBatch); //
    void setWeights(mynn::Mat weights); 
    void setBias(mynn::Mat bias);
    mynn::Mat getWeights();
    mynn::Mat getBias();

    double getL1WR() {return l1_weight_regularizer_;}
    double getL1BR() {return l1_bias_regularizer_;}
    double getL2WR() {return l2_weight_regularizer_;}
    double getL2BR() {return l2_bias_regularizer_;}

};