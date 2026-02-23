#include "MSE.h"

mynn::Mat MSE::forwardL(mynn::Mat predicted, mynn::Mat trueValue) {
    // what was i following,
    // in softmax true val inside the fn was 3*1
    // here true val is 3*3
    // E(yi - y)^2
    mynn::Mat losses(predicted.size().rows, 1);
    for(int i = 0; i < predicted.size().rows; ++i) {
        double mse = 0;
        for(int j = 0; j<predicted.size().cols; ++j) {
            double diff = (trueValue(i,j) - predicted(i,j));
            mse += diff * diff;
        }
        losses(i, 0) = mse/predicted.size().cols;
    }
    return losses;
    
}


mynn::Mat MSE::backwardL(mynn::Mat predicted, mynn::Mat trueValue) {
    // 2/n (y - yi), sometimes n is dropped;
    double n = predicted.size().cols;
    n = 2.0 / n;
    auto grad = predicted + trueValue * -1;
    grad = grad * n;
    return grad;
}