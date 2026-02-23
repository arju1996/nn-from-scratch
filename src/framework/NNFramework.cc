#include "NNFramework.h"

void NNFramework::add(std::unique_ptr<mynnLayer> layer) {
    layers_.push_back(std::move(layer));
}

void NNFramework::setLoss(std::unique_ptr<mynnLossFunction> loss) {
    loss_ = std::move(loss);
}

void NNFramework::setOptimizer(std::unique_ptr<mynnOptimizer> optimizer) {
    optimizer_ = std::move(optimizer);
}

mynn::Mat NNFramework::forward(const mynn::Mat& input) {
    
    if (!training_mode_) {
        mynn::Mat x = input;
        for(const auto& layer: layers_) {
            x = layer->forward(x);
        }
        return x;
    }

    graph_.caches.clear();
    graph_.caches.reserve(layers_.size());
    mynn::Mat x = input;
    for(const auto& layer: layers_) {
        LayerCache cache;
        x = layer->forward(x, &cache);
        graph_.caches.push_back(cache);
    }
    return x;
}


backwardResult NNFramework::backward(const mynn::Mat& dl_dz) {
    // backwardResult grads = input;
    auto l = graph_.caches.rbegin();
    auto grad = dl_dz;
    for(auto it = layers_.rbegin() ; it != layers_.rend(); ++it, ++l) {
        auto layer = it->get();
        auto grads = layer->backward(grad, *l);
        grad = grads.dl_dx.value();
        if(auto* dense = dynamic_cast<DenseLayer*>(layer);  dense && updateWeights_) {
            optimizer_->preUpdateParams();
            optimizer_->updateParams(*dense, grads.dl_dw.value(), grads.dl_db.value());
            optimizer_->postUpdateParams();
        }
    }
    return {grad, std::nullopt, std::nullopt};
}


void NNFramework::fit(const mynn::Mat& input, const mynn::Mat& output, int epoch, bool validation, const mynn::Mat& valinput, const mynn::Mat& valoutput) {
    for(int i = 0; i<epoch; ++i) {
        auto preds = forward(input);
        auto loss = loss_->forwardL(preds, output);
        
        if (i % 10 == 0) {
            std::cout<<"epoch = "<<i<<", loss="<<loss.meanof1d()<<"\n";
            if (validation) {
                eval(valinput, valoutput);
            }
        }
        
        
        auto dl_dz = loss_->backwardL(preds, output);
        auto grads = backward(dl_dz);
    }
}

void NNFramework::fit(const mynn::Dataset& train, int epoch, bool validation, const mynn::Dataset& test) {
    for(int i = 0; i<epoch; ++i) {
        auto preds = forward(train.input_);
        auto loss = loss_->forwardL(preds, train.output_);
        
        if (i % 10 == 0 || true) {
            std::cout<<"\ntraining epoch = "<<i<<", training loss = "<<loss.meanof1d()<<"\n";
            if (validation && true) {
                eval(test.input_, test.output_);
            }
        }
        
        
        auto dl_dz = loss_->backwardL(preds, train.output_);
        auto grads = backward(dl_dz);
    }
}


void NNFramework::eval(const mynn::Mat& input, const mynn::Mat& output) {
    enableTrainingMode(false);
    if(type_ == NNTaskType::REGRESSION) {
        auto preds = forward(input);

        auto loss = loss_->forwardL(preds, output);
        double accuracy_matrics_possibly_mse = loss.meanof1d();
        std::cout<<"validation; type = REGRESSION, ";
        std::cout<<"validation loss (mse or whatever regrtn) = "<<accuracy_matrics_possibly_mse<<"\n";
    } else if (type_ == NNTaskType::CLASSIFICATION) {
        
        // technically if last item is softmax, i dont need to pass it to softmax or even to loss;
        if(auto* mixlayer = dynamic_cast<A_Softmax_L_CatergoricalCrossEntropy*>(loss_.get())) {
            auto preds = forward(input);
            auto predicted = mixlayer->forwardwithoutLoss(preds);

            double accuracy = findAccuracyClassification(predicted, output);
            std::cout<<"type = CLASSIFICATION, layer = MIX, ";
            std::cout<<"accuracy for the testing data = "<<accuracy<<"\n";
            
            enableTrainingMode(true);
            return;
        }


        auto preds = forward(input);
        double accuracy = findAccuracyClassification(preds, output);
        std::cout<<"type = CLASSIFICATION, ";
        std::cout<<"accuracy for the testing data = "<<accuracy<<"\n";
    }

    enableTrainingMode(true);
}


double NNFramework::findAccuracyClassification(mynn::Mat y, mynn::Mat trueValues) {
    int correctCount = 0;
    for(int i = 0 ; i < y.size().rows; ++i) {
        std::pair<double, int> maxValue_Index = {0.0, -1};
        for (int j = 0; j <y.size().cols; ++j) {
            if(y(i, j) > maxValue_Index.first) {
                maxValue_Index.first = y(i, j);
                maxValue_Index.second = j;
            }
        }
        if(maxValue_Index.second == trueValues(i, 0))
            correctCount ++;
    }

    std::cout <<"correctCount = "<<correctCount<<" ";
    return correctCount*100.0/trueValues.size().rows;
}