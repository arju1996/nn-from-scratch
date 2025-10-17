// #include "Softmax.h"
// #include "LossFunction.h"

#include "A_Softmax_L_CatergoricalCrossEntropy.h"

// class ActivationSoftmax_LossCatergoricalCrossEntropy {
//     Softmax activation;
//     CategoricalCrossEntropyL loss;
//     public:
//     mynn::Mat forward(mynn::Mat input, mynn::Mat trueValues);
//     void backward(mynn::Mat predicted, mynn::Mat trueValues);
// };

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
