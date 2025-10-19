#include "Mat.h"

class DropoutLayer {
    double success_rate_;
    mynn::Mat binary_mask_;
    public:
    DropoutLayer(double rate);
    mynn::Mat forward(mynn::Mat input);
    mynn::Mat backward(mynn::Mat input);
};