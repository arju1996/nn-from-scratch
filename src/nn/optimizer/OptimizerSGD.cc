#include "OptimizerSGD.h"


OptimizerSGD::OptimizerSGD() : 
    learning_rate_(1),
    current_learning_rate_(1),
    decay_(0.001),
    use_decay_(true),
    iteration_(0)
    {
        // int i = 0
}

OptimizerSGD::OptimizerSGD(
    double learning_rate,
    double decay,
    bool use_decay,
    double momentum_factor,
    bool use_momentum
) : 
    learning_rate_(learning_rate),
    current_learning_rate_(learning_rate),
    decay_(decay),
    use_decay_(use_decay),
    momentum_factor_(momentum_factor),
    use_momentum_(use_momentum),
    iteration_(0)
    {
        // int i = 0;
}

void OptimizerSGD::preUpdateParams() {
    current_learning_rate_ = learning_rate_ / (1 + decay_ * iteration_);
}


void OptimizerSGD::updateParams(DenseLayer& layer, mynn::Mat dl_dw, mynn::Mat dl_db) {
    // transpose dl_dw to match weight shape (derived from scratch)
    dl_dw = dl_dw.transpose();

    auto weight = layer.getWeights();
    auto bias = layer.getBias();

    double effectiveLR;
    if(use_decay_) {
        effectiveLR = current_learning_rate_;
    } else {
        effectiveLR = learning_rate_;
    }

    mynn::Size weightSize = weight.size();
    mynn::Size biasSize = bias.size();
    mynn::Mat weight_updates;
    mynn::Mat bias_updates;
    if(use_momentum_) {
        // if layer doesnt have momentum set it to 0,0,0
        auto it = weight_momentum_.find(&layer);
        if(it == weight_momentum_.end()) {
            weight_momentum_[&layer] = mynn::Mat(weightSize.rows, weightSize.cols);
        }
        auto& layerWeight = weight_momentum_[&layer];

        auto it2 = bias_momentum_.find(&layer);
        if(it2 == bias_momentum_.end()) {
            bias_momentum_[&layer] = mynn::Mat(biasSize.rows, biasSize.cols);
        }
        auto& layerBias = bias_momentum_[&layer];

        layerWeight =  layerWeight * momentum_factor_ + dl_dw * -effectiveLR;
        layerBias =  layerBias * momentum_factor_ + dl_db * -effectiveLR;

        weight_updates = layerWeight;
        bias_updates = layerBias;
    } else {
        weight_updates = (dl_dw * -effectiveLR);
        bias_updates = (dl_db * -effectiveLR);
    }

    weight = weight + weight_updates;
    bias = bias + bias_updates;

    layer.setWeights(weight);
    layer.setBias(bias);

}

void OptimizerSGD::postUpdateParams() {
    iteration_ = iteration_ + 1;
}
