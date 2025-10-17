#pragma once
// #include "Softmax.h"
// #include "LossFunction.h"

#include "activation/Softmax.h"
#include "loss/LossFunction.h"

#include <tuple>


class A_Softmax_L_CatergoricalCrossEntropy {
    Softmax activation_;
    CategoricalCrossEntropyL loss_;
    public:
    std::tuple<mynn::Mat, mynn::Mat> forward(mynn::Mat input, mynn::Mat trueValues);
    mynn::Mat backward(mynn::Mat predicted, mynn::Mat trueValues);
};