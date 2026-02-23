// #include "Softmax.h"
// #include "LossFunction.h"

#include "A_Softmax_L_CatergoricalCrossEntropy.h"

std::tuple<mynn::Mat, mynn::Mat> A_Softmax_L_CatergoricalCrossEntropy::forward(mynn::Mat input, mynn::Mat trueValues) {
    //
    auto predicted = activation_.forward(input);
    auto loss = loss_.forward(predicted, trueValues);
    return {predicted, loss};
}

mynn::Mat A_Softmax_L_CatergoricalCrossEntropy::backward(mynn::Mat predicted, mynn::Mat trueValues) {
    // loss = predicted - trueval;


    int samples = predicted.size().rows;
    if(trueValues.size().cols != 1) {
        throw std::runtime_error("one hot encoding isnt supported now");
    }
    auto loss = predicted;
    for(int i = 0, r = predicted.size().rows; i<r; ++i) {
        loss(i, trueValues(i, 0)) -= 1;
        for(int j = 0, c = predicted.size().cols; j<c; ++j) {
            loss(i, j) /= samples;
        }
    }

    return loss;
}

double A_Softmax_L_CatergoricalCrossEntropy::regularizationLoss(DenseLayer* layer) {
    return loss_.regularizationLoss(layer);
}

mynn::Mat A_Softmax_L_CatergoricalCrossEntropy::forwardL(mynn::Mat input, mynn::Mat trueValue) {
    auto predicted = activation_.forward(input);
    auto loss = loss_.forward(predicted, trueValue);
    return loss;
}


mynn::Mat A_Softmax_L_CatergoricalCrossEntropy::backwardL(mynn::Mat predicted, mynn::Mat trueValue) {
    return backward(predicted, trueValue);   
}

mynn::Mat A_Softmax_L_CatergoricalCrossEntropy::forwardwithoutLoss(mynn::Mat input) {
    auto predicted = activation_.forward(input);
    return predicted;
}
