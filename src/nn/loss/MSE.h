#pragma once

#include "layer.h"

class MSE : public mynnLossFunction {
    public:
    mynn::Mat forwardL(mynn::Mat predicted, mynn::Mat trueValue) override;
    mynn::Mat backwardL(mynn::Mat predicted, mynn::Mat trueValue) override;
};