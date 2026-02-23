#pragma once

#include <vector>
#include <cassert>

#include "Mat.h"

#include "layer.h"


class Softmax : public mynnLayer{
public:
    std::vector<double> run(std::vector<double> input);
    std::vector<std::vector<double>> runInputBatch(std::vector<std::vector<double>> inputBatch);

    mynn::Mat forward(mynn::Mat input);


    mynn::Mat forward(const mynn::Mat& input, LayerCache* cache) override;
    backwardResult backward(const mynn::Mat& dl_dz, LayerCache& cache) override;
};