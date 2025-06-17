#include <iostream>
#include <array>
#include <vector>

#include <cassert>

#include "random.h"

#include "DenseLayer.h"

DenseLayer::DenseLayer(int input_size, int no_of_nodes) : input_size_(input_size), no_of_nodes_(no_of_nodes) {
    // set random weight to each of the nodes (of size input_size);
    for(int i = 0 ; i < no_of_nodes_; ++i) {
        weights_.push_back(math::random::GetRandomNormVector(0, 1, input_size_));
        // scaling the weights down, i dont know why.
        for(auto& item: weights_.back())
            item /= 100;
    }

    // set random bias for each nodes
    // in the original code this was set to 0, but lemme keep this as randoms to know why
    // they kept it as 0;
    bias_ = math::random::GetRandomNormVector(0, 1, no_of_nodes_);
}

void DenseLayer::PrintDenseLayer() {
    std::cout<<"hello this is the layer \n weights are \n";
    for(auto weight: weights_) {
        for(auto item: weight) {
            std::cout<<item<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\nbiases are\n";
    for(auto item: bias_) {
        std::cout<<item<<" ";
    }
    std::cout<<"\n";
}

void DenseLayer::whatever(){}

inline double DenseLayer::VectorDotVector(std::vector<double> a, std::vector<double> b) {
    assert(a.size() == b.size());
    double result = 0;
    for(int i = 0; i < a.size(); ++i) {
        result += a.at(i) * b.at(i);
    }
    return result;
}

std::vector<double> DenseLayer::run(std::vector<double> input) {
    // run the input on each nodes
    std::vector<double> result;
    for(int i = 0; i < no_of_nodes_; ++i) {
        result.push_back(VectorDotVector(weights_.at(i), input) + bias_.at(i));
    }
    return result;
}

std::vector<std::vector<double>> DenseLayer::runInputBatch(std::vector<std::vector<double>> inputBatch) {
    std::vector<std::vector<double>> result;
    // run the each inputs on the layer
    for(int i = 0; i < inputBatch.size(); ++i) {
        result.push_back(run(inputBatch.at(i)));
    }
    return result;
}


