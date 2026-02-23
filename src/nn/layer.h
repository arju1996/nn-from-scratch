#pragma once

#include "Mat.h"
#include <optional>
// #include "layers/DenseLayer.h"

class DenseLayer;

enum NNTaskType {
    REGRESSION,
    CLASSIFICATION
};

class mynnOptimizer {
    public:
    virtual void preUpdateParams() = 0;
    virtual void updateParams(DenseLayer& layer, mynn::Mat dl_dw, mynn::Mat dl_db) = 0;
    virtual void postUpdateParams() = 0;
};


class mynnLossFunction {
    public:
    virtual mynn::Mat forwardL(mynn::Mat predicted, mynn::Mat trueValue) = 0;
    virtual mynn::Mat backwardL(mynn::Mat predicted, mynn::Mat trueValue) = 0;

    // combined forward should return 2 Mats
};



// i hope one layer only need to cache one data
struct LayerCache {mynn::Mat data;};

struct ComputationGraph {
    // for the framework
    std::vector<LayerCache> caches;  // One per layer
};

struct backwardResult {
    std::optional<mynn::Mat> dl_dx;
    std::optional<mynn::Mat> dl_dw;
    std::optional<mynn::Mat> dl_db;
};

class mynnLayer {
    public:
    virtual mynn::Mat forward(const mynn::Mat& input, LayerCache* cache = nullptr) = 0;
    virtual backwardResult backward(const mynn::Mat& dl_dz, LayerCache& cache) = 0;
    virtual bool isParametric() const { return false; };
};


// class ParametricLayer : public mynnLayer {
//     virtual bool isParametric() const { return true; };
//     // weights amd biases
//     virtual void getParams(std::vector<mynn::Mat>& params) = 0;
//     // all the grads
//     virtual void getGradients(std::vector<mynn::Mat>& grads) = 0;
// };



// #include <vector>
// #include <memory>

// class NNFramework {
//     std::vector<std::unique_ptr<mynnLayer>> layers_;
//     std::unique_ptr<mynnLossFunction> loss_;
//     std::unique_ptr<mynnOptimizer> optimizer_;

//     public:
//     void add(std::unique_ptr<mynnLayer> layer);
//     mynn::Mat forward(mynn::Mat input);
//     void backward(mynn::Mat dl_dy);
//     // void fit(inputData input, groundtruth output);
//     // void eval();
// };

