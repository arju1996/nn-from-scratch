#pragma once
// #include "Softmax.h"
// #include "LossFunction.h"

#include "activation/Softmax.h"
#include "loss/LossFunction.h"
#include "layer.h"


#include <tuple>


class A_Softmax_L_CatergoricalCrossEntropy : public mynnLossFunction {
    Softmax activation_;
    CategoricalCrossEntropyL loss_;
    public:
    std::tuple<mynn::Mat, mynn::Mat> forward(mynn::Mat input, mynn::Mat trueValues);
    mynn::Mat backward(mynn::Mat predicted, mynn::Mat trueValues);
    double regularizationLoss(DenseLayer* layer);
    mynn::Mat forwardL(mynn::Mat predicted, mynn::Mat trueValue) override;
    mynn::Mat backwardL(mynn::Mat predicted, mynn::Mat trueValue) override;

    mynn::Mat forwardwithoutLoss(mynn::Mat input);

};