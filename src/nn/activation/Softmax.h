#pragma once

#include <vector>
#include <cassert>

#include "Mat.h"


class Softmax {
public:
    std::vector<double> run(std::vector<double> input);
    std::vector<std::vector<double>> runInputBatch(std::vector<std::vector<double>> inputBatch);

    mynn::Mat forward(mynn::Mat input);
};