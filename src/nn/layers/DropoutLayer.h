#include "Mat.h"
#include "layer.h"


class DropoutLayer : public mynnLayer {
    double success_rate_;
    mynn::Mat binary_mask_;
    public:
    DropoutLayer(double rate);
    mynn::Mat forwardold(mynn::Mat input);
    mynn::Mat backward(mynn::Mat input);

    mynn::Mat forward(const mynn::Mat& input, LayerCache* cache = nullptr) override;
    backwardResult backward(const mynn::Mat& dl_dz, LayerCache& cache) override;
};