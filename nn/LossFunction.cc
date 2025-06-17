#include "LossFunction.h"
#include <iostream>
#include <algorithm>


void CategoricalCrossEntropy::clip(double &x, double lower, double upper) {
    x = std::max(lower, std::min(x, upper));
}

void CategoricalCrossEntropy::clip(
    std::vector<std::vector<double>>& data,
    double lower,
    double upper
) {
    for(auto &a : data) {
        for(auto &b: a) {
            clip(b, lower, upper);
        }
    }
}


// case when outputs are labels, ie not one hot encoded;
std::vector<double> CategoricalCrossEntropy::forward(
    std::vector<std::vector<double>> resultafteractivationforabatch ,
    std::vector<int> classTargets
) {
    
    // clipping is kinda optional
    double lower = std::exp(-7);
    double upper = std::exp(7);
    clip(resultafteractivationforabatch, lower, upper);
    //

    std::vector<double> losses;
    for(int i = 0; i< resultafteractivationforabatch.size(); ++i) {
        auto softmax = resultafteractivationforabatch[i];
        assert(classTargets[i]<softmax.size());
        losses.push_back(0-std::log(softmax.at(classTargets[i])));
    }
    return losses;
}

// case when outputs are one hot encoded
void CategoricalCrossEntropy::calculate(
    std::vector<std::vector<double>> resultafteractivationforabatch ,
    std::vector<std::vector<int>> classTargets
) {
    int b;
}


double LossFunction::calculate(
    std::vector<std::vector<double>> output,
    std::vector<int> y
) {

    assert(output.size(), y.size());

    std::vector<double> losses = forward(output, y);
    double avg_loss = 0.0;
    for(auto &loss: losses)
        avg_loss += loss;
    
    avg_loss /= y.size();
    return avg_loss;
}