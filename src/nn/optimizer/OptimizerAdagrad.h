#include "layers/DenseLayer.h"

#include <unordered_map>
#include "layer.h"

class OptimizerADAGRAD : public mynnOptimizer {
    double learning_rate_;
    double current_learning_rate_;
    double decay_;
    int iteration_;
    bool use_decay_;
    const double eps_;

    // making this class statfull
    std::unordered_map<DenseLayer*, mynn::Mat> weight_cache_;
    std::unordered_map<DenseLayer*, mynn::Mat> bias_cache_;

    public:
    OptimizerADAGRAD(double learning_rate, double decay, bool use_decay, double epsilon);
    void preUpdateParams() override;
    void updateParams(DenseLayer& layer, mynn::Mat dl_dw, mynn::Mat dl_db) override;
    void postUpdateParams() override;

    double getCurrentLearningRate() const {
        return current_learning_rate_;
    }
};