#include "OptimizerRMSprop.h"


// OptimizerADAGRAD::OptimizerSGD() : 
//     learning_rate_(1),
//     current_learning_rate_(1),
//     decay_(0.001),
//     use_decay_(true),
//     iteration_(0)
//     {
//         // int i = 0
// }

OptimizerRMSprop::OptimizerRMSprop(
    double learning_rate,
    double decay,
    bool use_decay,
    double epsilon,
    double rho
) : 
    learning_rate_(learning_rate),
    current_learning_rate_(learning_rate),
    decay_(decay),
    use_decay_(use_decay),
    iteration_(0),
    eps_(epsilon),
    rho_(rho)
    {
        // int i = 0;
}

void OptimizerRMSprop::preUpdateParams() {
    current_learning_rate_ = learning_rate_ / (1 + decay_ * iteration_);
}


void OptimizerRMSprop::updateParams(DenseLayer& layer, mynn::Mat dl_dw, mynn::Mat dl_db) {
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

    auto it = weight_cache_.find(&layer);
    if(it == weight_cache_.end()) {
        weight_cache_[&layer] = mynn::Mat(weightSize.rows, weightSize.cols);
    }
    auto& layerWeightCache = weight_cache_[&layer];
    auto it2 = bias_cache_.find(&layer);
    if(it2 == bias_cache_.end()) {
        bias_cache_[&layer] = mynn::Mat(biasSize.rows, biasSize.cols);
    }
    auto& layerBiasCache = bias_cache_[&layer];

    layerWeightCache = (layerWeightCache * rho_) + ( dl_dw.sq() * (1-rho_));
    layerBiasCache = (layerBiasCache * rho_) + ( dl_db.sq() * (1-rho_));

    mynn::Mat wdenom = (layerWeightCache + eps_ ).sqrt();
    mynn::Mat bdenom = (layerBiasCache + eps_).sqrt();

    // he is following this equation
    // mynn::Mat wdenom = (layerWeightCache  ).sqrt() + eps_;
    // mynn::Mat bdenom = (layerBiasCache ).sqrt()+ eps_;
    
    // for SGD denominators will be 1 (the one without the momentum, but just the decay)
    auto weight_update = (dl_dw * -effectiveLR) / wdenom;
    auto bias_update = (dl_db * -effectiveLR) / bdenom;

    weight = weight + weight_update;
    bias = bias + bias_update;

    layer.setWeights(weight);
    layer.setBias(bias);
}

void OptimizerRMSprop::postUpdateParams() {
    iteration_ = iteration_ + 1;
}
