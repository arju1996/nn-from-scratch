#include <iostream>
#include <array>
#include <vector>

#include <cassert>

#include "random.h"

#include "DenseLayer.h"

DenseLayer::DenseLayer(int input_size, int no_of_nodes) : input_size_(input_size), no_of_nodes_(no_of_nodes) {
    // set random weight to each of the nodes (of size input_size);
    
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

    for(int i = 0 ; i < no_of_nodes_; ++i) {
        weights.push_back(math::random::GetRandomNormVector(0, 1, input_size_));
        // scaling the weights down, i dont know why.
        for(auto& item: weights.back())
            item /= 100;
    }

    weights_ = mynn::Mat(weights);

    // set random bias for each nodes
    // in the original code this was set to 0, but lemme keep this as randoms to know why
    // they kept it as 0;
    bias = math::random::GetRandomNormVector(0, 1, no_of_nodes_);
    // bias_ = mynn::Mat( 1, no_of_nodes_, bias);
    bias_ = mynn::Mat( 1, no_of_nodes_);
}

DenseLayer::DenseLayer(int input_size, int no_of_nodes,double l1_weight_regularizer,double l1_bias_regularizer, double l2_weight_regularizer, double l2_bias_regularizer)
 : input_size_(input_size), 
 no_of_nodes_(no_of_nodes),
 l1_weight_regularizer_(l1_weight_regularizer),
 l1_bias_regularizer_(l1_bias_regularizer),
 l2_weight_regularizer_(l2_weight_regularizer),
 l2_bias_regularizer_(l2_bias_regularizer)
 {
    // set random weight to each of the nodes (of size input_size);
    
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;

    for(int i = 0 ; i < no_of_nodes_; ++i) {
        weights.push_back(math::random::GetRandomNormVector(0, 1, input_size_));
        // scaling the weights down, i dont know why.
        for(auto& item: weights.back())
            item /= 100;
    }

    weights_ = mynn::Mat(weights);

    // set random bias for each nodes
    // in the original code this was set to 0, but lemme keep this as randoms to know why
    // they kept it as 0;
    bias = math::random::GetRandomNormVector(0, 1, no_of_nodes_);
    // bias_ = mynn::Mat( 1, no_of_nodes_, bias);
    bias_ = mynn::Mat( 1, no_of_nodes_);
}


void DenseLayer::setWeights(mynn::Mat weights) {
    weights_ = weights;
}
void DenseLayer::setBias(mynn::Mat bias) {
   bias_ = bias;
}
mynn::Mat DenseLayer::getWeights() {
    return weights_;
}
mynn::Mat DenseLayer::getBias() {
    return bias_;
}

void DenseLayer::PrintDenseLayer() {
    std::cout<<"hello this is the layer \n weights are \n";
    std::cout << weights_;
    /*
    for(auto weight: weights_) {
        for(auto item: weight) {
            std::cout<<item<<" ";
        }
        std::cout<<"\n";
    }
    */
    std::cout<<"\nbiases are\n";
    std::cout << bias_;
    /*
    for(auto item: bias_) {
        std::cout<<item<<" ";
    }
    std::cout<<"\n";
    */
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
        // result.push_back(VectorDotVector(weights_.at(i), input) + bias_.at(i));
    }
    return result;
}

mynn::Mat DenseLayer::forward(mynn::Mat input) {
    // mynn::Mat result = weights_ * input + bias_;
    // mynn::Mat result = weights_.multiply(input) + bias_;
    // mynn::Mat result = input.multiply(weights_) + bias_;
    // mynn::Mat result = weights_.transpose().multiply(input) + bias_;
    mynn::Mat result = input.multiply(weights_.transpose()) + bias_;


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

std::tuple<mynn::Mat, mynn::Mat, mynn::Mat> DenseLayer::backward(mynn::Mat dl_dz, mynn::Mat input) {
    // find dl_dw, dl_db, dl_dx

    // std::vector<double> dl_dw = xt * dl_dz;
    // std::vector<double> dl_db = dl_dz;
    // std::vector<double> dl_dx = dl_dz * weights_;

    // mynn::Mat input;
    // assume input has values
    mynn::Size inputsize = input.size();

    mynn::Mat dl_dw = input.transpose().multiply(dl_dz);
    mynn::Mat dl_db(1, dl_dz.size().cols);
    // adds the dl_dz from each batch ie z11 + z21 + z31 (z of same neuron from each batch)
    for(int i = 0,c = dl_dz.size().cols; i < c; ++i) {
        double sum = 0;
        for(int j = 0,r = dl_dz.size().rows; j < r; ++j) {
            sum += dl_dz(j, i);
        }
        dl_db(0, i) = sum;
    }
    mynn::Mat dl_dx = dl_dz.multiply(weights_);

    // regularizer,
    dl_dw = dl_dw.transpose();
    if(l1_weight_regularizer_ > 0) {
        mynn::Mat dl_dwregu(weights_.size().rows, weights_.size().cols);
        for(int i = 0,c = weights_.size().rows; i < c; ++i) {
            for(int j = 0,r = weights_.size().cols; j < r; ++j) {
                dl_dwregu(i,j) = weights_(i , j) >= 0 ? l1_weight_regularizer_ : -l1_weight_regularizer_;
            }
        }
        dl_dw = dl_dw + dl_dwregu;
    }
    if(l2_weight_regularizer_ > 0) {
        mynn::Mat dl_dwregu(weights_.size().rows, weights_.size().cols);
        dl_dwregu = weights_ * (2 * l2_weight_regularizer_);
        dl_dw = dl_dw + dl_dwregu;
    }
    if(l1_bias_regularizer_ > 0) {
        mynn::Mat dl_dbregu(bias_.size().rows, bias_.size().cols);
        for(int i = 0,c = bias_.size().rows; i < c; ++i) {
            for(int j = 0,r = bias_.size().cols; j < r; ++j) {
                dl_dbregu(i,j) = bias_(i , j) >= 0 ? l1_bias_regularizer_ : -l1_bias_regularizer_;
            }
        }
        dl_db = dl_db + dl_dbregu;
    }
    if(l2_bias_regularizer_ > 0) {
        mynn::Mat dl_dbregu(dl_db.size().rows, dl_db.size().cols);
        dl_dbregu = bias_ * (2 * l2_bias_regularizer_);
        dl_db = dl_db + dl_dbregu;
    }
    dl_dw = dl_dw.transpose();

    std::tuple<mynn::Mat, mynn::Mat, mynn::Mat> result = {dl_dw, dl_db, dl_dx};
    return result;
}

void DenseLayer::backwardInputBatch(std::vector<std::vector<double>> dl_dz) {
    // std::vector<double> dl_dw = xt * dl_dz;
    // std::vector<double> dl_db = dl_dz; // raw sum, ie dl_db[i] = sum(dl_dz_tranpose[i])
    // std::vector<double> dl_dx = dl_dz * weights_;
}

