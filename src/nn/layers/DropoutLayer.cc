#include "DropoutLayer.h"

#include "random.h"


DropoutLayer::DropoutLayer(double rate) : success_rate_(1-rate) {

}

mynn::Mat DropoutLayer::forward(mynn::Mat input) {
    std::random_device rd;
    // unsigned int seed = 234;
    unsigned int seed = rd();
    binary_mask_ = mynn::Mat(math::random::GetBernoulliSamples(success_rate_, seed, input.size().rows, input.size().cols)) * (1/success_rate_);
    mynn::Mat output = input.multiplyElementWise(binary_mask_);
    // std::cout<<"binary_mask_ = "<<binary_mask_<<"\n";
    return output;
}

mynn::Mat DropoutLayer::backward(mynn::Mat dvalues) {
    return dvalues.multiplyElementWise(binary_mask_);
}