#include "OptimizerADAM.h"

OptimizerADAM::OptimizerADAM(
    double learning_rate,
    double decay,
    bool use_decay,
    double epsilon,
    // double rho,
    double b1,
    double b2
) : 
    learning_rate_(learning_rate),
    current_learning_rate_(learning_rate),
    decay_(decay),
    use_decay_(use_decay),
    iteration_(0),
    eps_(epsilon),
    // rho_(rho),
    b1_(b1),
    b2_(b2)
    {
        // int i = 0;
}

void OptimizerADAM::preUpdateParams() {
    current_learning_rate_ = learning_rate_ / (1 + decay_ * iteration_);
}


void OptimizerADAM::updateParams(DenseLayer& layer, mynn::Mat dl_dw, mynn::Mat dl_db) {
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

    // cache
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
    // momentum
    auto it3 = weight_momentum_.find(&layer);
    if(it3 == weight_momentum_.end()) {
        weight_momentum_[&layer] = mynn::Mat(weightSize.rows, weightSize.cols);
    }
    auto& layerWeightMomentum = weight_momentum_[&layer];

    auto it4 = bias_momentum_.find(&layer);
    if(it4 == bias_momentum_.end()) {
        bias_momentum_[&layer] = mynn::Mat(biasSize.rows, biasSize.cols);
    }
    auto& layerBiasMomentum = bias_momentum_[&layer];

    // momentum
    layerWeightMomentum =  layerWeightMomentum * b1_ + dl_dw * (1-b1_);
    layerBiasMomentum =  layerBiasMomentum * b1_ + dl_db * (1-b1_);

    double MomentumBalancingFactor = (1 - pow(b1_, this->iteration_ + 1));
    MomentumBalancingFactor = 1/MomentumBalancingFactor;
    auto layerWeightMomentumCorrected = layerWeightMomentum * MomentumBalancingFactor;
    auto layerBiasMomentumCorrected = layerBiasMomentum * MomentumBalancingFactor;


    // cache
    double cacheBalancingbalancingFactor = 1 - pow(b2_, this->iteration_ + 1);
    cacheBalancingbalancingFactor = 1/cacheBalancingbalancingFactor;


    layerWeightCache = (layerWeightCache * b2_) + ( dl_dw.sq() * (1-b2_));
    layerBiasCache = (layerBiasCache * b2_) + ( dl_db.sq() * (1-b2_));
    auto layerWeightCacheCorrected = layerWeightCache * cacheBalancingbalancingFactor;
    auto layerBiasCacheCorrected = layerBiasCache * cacheBalancingbalancingFactor;


    mynn::Mat wdenom = (layerWeightCacheCorrected + eps_ ).sqrt();
    mynn::Mat bdenom = (layerBiasCacheCorrected + eps_).sqrt();

    // he is following this equation
    // mynn::Mat wdenom = (layerWeightCacheCorrected  ).sqrt() + eps_;
    // mynn::Mat bdenom = (layerBiasCacheCorrected ).sqrt()+ eps_;
    
    // for SGD denominators will be 1 (the one without the momentum, but just the decay)
    auto weight_update = (layerWeightMomentumCorrected * -effectiveLR) / wdenom;
    auto bias_update = (layerBiasMomentumCorrected * -effectiveLR) / bdenom;

    weight = weight + weight_update;
    bias = bias + bias_update;

    layer.setWeights(weight);
    layer.setBias(bias);
}

void OptimizerADAM::postUpdateParams() {
    iteration_ = iteration_ + 1;
}
