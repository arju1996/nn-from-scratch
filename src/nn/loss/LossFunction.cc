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

    assert(output.size() == y.size());

    std::vector<double> losses = forward(output, y);
    double avg_loss = 0.0;
    for(auto &loss: losses)
        avg_loss += loss;
    
    avg_loss /= y.size();
    return avg_loss;
}

mynn::Mat CategoricalCrossEntropyL::forward(mynn::Mat predicted, mynn::Mat trueValues) {
    // make trueValues to a matrix;
    // loss = - truevalue * log predic

    if(trueValues.size().cols != 1) {
        throw std::runtime_error("one hot encoding isnt supported now");
    }
    
    double lower = std::exp(-7);
    double upper = std::exp(7);
    clip(predicted, lower, upper);
    //
    // somehow losses was a row matrix before, ive changed it to col matrix now;
    // because true values was a column matrix
    mynn::Mat losses (trueValues.size().rows, 1);
    for(int i = 0, r = trueValues.size().rows; i<r; ++i) {
        losses(i, 0) = 0-std::log(predicted(i, trueValues(i, 0)));
    }
    return losses;
}


void CategoricalCrossEntropyL::backward(mynn::Mat predicted, mynn::Mat trueValues) {

    if(trueValues.size().rows != 1) {
        throw std::runtime_error("one hot encoding isnt supported now");
    }

    int sample = predicted.size().cols;
    int labels = trueValues.size().cols;

    // l1 = - [y11 log(y11) + y12 log(y12) + y13 log(y13) ]
    // mynn::Mat dl_dy = predicted / truevalues ?
    mynn::Mat dl_dy (predicted.size().rows, predicted.size().cols);
    for(int i = 0, r = predicted.size().rows; i<r; ++i) {
        dl_dy(i, trueValues(0, i)) = 1/dl_dy(i, trueValues(0, i));

        // optional ie normalize
        dl_dy(i, trueValues(0, i)) /= sample;
    }
    
}



void CategoricalCrossEntropyL::clip(double &x, double lower, double upper) {
    x = std::max(lower, std::min(x, upper));
}

void CategoricalCrossEntropyL::clip(
    mynn::Mat& data,
    double lower,
    double upper
) {

    for(int i = 0, r = data.size().rows; i<r; ++i) {
        for(int j = 0, c = data.size().cols; j<c; ++j) {
            clip(data(i, j), lower, upper);
        }
    }
}

double CategoricalCrossEntropyL::regularizationLoss(DenseLayer* layer) {
    double regularizationLoss = 0;
    if(layer->getL1WR() > 0) {
        double l1sum = 0;
        auto w = layer->getWeights();
        for(int i = 0, r = w.size().rows; i<r; ++i) {
            for(int j = 0, c = w.size().cols; j<c; ++j) {
                l1sum += abs(w(i, j)); 
            }
        }
        regularizationLoss += layer->getL1WR() * l1sum;
    }
    if(layer->getL1BR() > 0) {
        double l1sum = 0;
        auto b = layer->getBias();
        for(int i = 0, r = b.size().rows; i<r; ++i) {
            for(int j = 0, c = b.size().cols; j<c; ++j) {
                l1sum += abs(b(i, j)); 
            }
        }
        regularizationLoss += layer->getL1BR() * l1sum;
    }
    if(layer->getL2WR() > 0) {
        double l2sum = 0;
        auto w = layer->getWeights();
        for(int i = 0, r = w.size().rows; i<r; ++i) {
            for(int j = 0, c = w.size().cols; j<c; ++j) {
                l2sum += w(i, j) * w(i, j); 
            }
        }
        regularizationLoss += layer->getL1WR() * l2sum;
    }
    if(layer->getL2BR() > 0) {
        double l2sum = 0;
        auto b = layer->getBias();
        for(int i = 0, r = b.size().rows; i<r; ++i) {
            for(int j = 0, c = b.size().cols; j<c; ++j) {
                l2sum += b(i, j) * b(i, j); 
            }
        }
        regularizationLoss += layer->getL2BR() * l2sum;
    }

    return regularizationLoss;
}
