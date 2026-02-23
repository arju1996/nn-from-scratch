#include "Softmax.h"
#include <algorithm>
#include <cmath>

std::vector<double> Softmax::run(std::vector<double> input) {
    std::vector<double> output;
    std::vector<double> exponents;
    output.reserve(input.size());
    exponents.reserve(input.size());

    // find min
    auto it = std::max_element(input.begin(), input.end());
    assert(it != input.end());
    double max = *it;

    // sub max and exp
    // couldnt it be better if i subtract mean instead of max?, ie to avoid larger nos
    double e = std::exp(1.0);
    double sumofexp = 0.0;
    for(auto item: input) {
        double power = std::pow(e, item - max);
        exponents.push_back(power);
        sumofexp += power;
    }

    for(auto& item: exponents) {
        item /= sumofexp;
    }

    return exponents;
}

mynn::Mat Softmax::forward(mynn::Mat input) {
    mynn::Mat output(input.size().rows, input.size().cols);
    // for a 1d array, first find the e ^ ycap11 + e ^ ycap12 + e ^ ycap13
    // then devide each item on the row by the sum;
    for(int i = 0, r = input.size().rows; i < r; ++i) {
        // find biggest, find sum of ex;
        // TODO: VALIDATE
        // this could be an issue actually, ie max cannot be zero, what if all negs 
        double max = 0;
        for(int j = 0, c = input.size().cols; j < c; ++j) {
            if(input(i,j)>max)
                max = input(i,j);
        }

        // find e^y11 + e^y12 + ..
        double e = std::exp(1.0);
        double sumofexp = 0.0;
        for(int j = 0, c = input.size().cols; j < c; ++j) {
            sumofexp += std::pow(e, input(i,j) - max);
        }

        for(int j = 0, c = input.size().cols; j < c; ++j) {
            output(i,j) = std::pow(e, input(i,j) - max) / sumofexp;
        }
    }
    return output;
}

std::vector<std::vector<double>> Softmax::runInputBatch(std::vector<std::vector<double>> inputBatch) {
    std::vector<std::vector<double>> output;
    for(auto input: inputBatch) {
        output.push_back(run(input));
    }
    return output;
}

mynn::Mat Softmax::forward(const mynn::Mat& input, LayerCache* cache) {
    if (cache) {
        
    }
    return forward(input);
}

backwardResult Softmax::backward(const mynn::Mat& dl_da, LayerCache& cache) {
    return {
        std::nullopt,
        std::nullopt,
        std::nullopt
    };
}