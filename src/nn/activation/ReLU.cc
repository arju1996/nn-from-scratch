#include "ReLU.h"

ReLU::ReLU() {

}

std::vector<double> ReLU::run(std::vector<double> input) {
    std::vector<double> output;
    output.reserve(input.size());
    for(auto item: input) {
        output.push_back(std::max(0.0, item));
    }
    return output;
}

std::vector<std::vector<double>> ReLU::runInputBatch(std::vector<std::vector<double>> inputBatch) {
    std::vector<std::vector<double>> output;
    for(auto input: inputBatch) {
        output.push_back(run(input));
    }
    return output;
}