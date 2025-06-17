#include "Softmax.h"
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

std::vector<std::vector<double>> Softmax::runInputBatch(std::vector<std::vector<double>> inputBatch) {
    std::vector<std::vector<double>> output;
    for(auto input: inputBatch) {
        output.push_back(run(input));
    }
    return output;
}