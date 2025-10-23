#pragma once

#include "Mat.h"


namespace mynn {
    namespace fast {
        mynn::Mat dense_forward(const mynn::Mat& input, const mynn::Mat& weight, const mynn::Mat& bias);
        mynn::Mat relu_forward(const mynn::Mat& input);
        mynn::Mat drop_forward(const mynn::Mat& input, const mynn::Mat& mask);
        mynn::Mat softmax_forward(const mynn::Mat& input);
    }
}