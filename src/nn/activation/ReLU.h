#include <vector>
#include <cassert>

#include "Mat.h"
#include "layer.h"


class ReLU : public mynnLayer {
public:
    ReLU();
    mynn::Mat run(const mynn::Mat& input);
    mynn::Mat backward(mynn::Mat dl_da, mynn::Mat input);

    mynn::Mat forward(const mynn::Mat& input, LayerCache* cache = nullptr) override;
    backwardResult backward(const mynn::Mat& dl_dz, LayerCache& cache) override;
};