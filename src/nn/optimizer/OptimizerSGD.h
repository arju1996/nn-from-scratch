#include "layers/DenseLayer.h"

#include <unordered_map>
#include "layer.h"
class OptimizerSGD : public mynnOptimizer {
    double learning_rate_;
    double current_learning_rate_;
    double decay_;
    double momentum_factor_;
    int iteration_;
    bool use_decay_;
    bool use_momentum_;

    // making this class statfull
    std::unordered_map<DenseLayer*, mynn::Mat> weight_momentum_;
    std::unordered_map<DenseLayer*, mynn::Mat> bias_momentum_;

    public:
    OptimizerSGD();
    OptimizerSGD(double learning_rate, double decay, bool use_decay, double momentum_factor, bool use_momentum);
    void preUpdateParams() override;
    void updateParams(DenseLayer& layer, mynn::Mat dl_dw, mynn::Mat dl_db) override;
    void postUpdateParams() override;

    double getCurrentLearningRate() const {
        return current_learning_rate_;
    }
};