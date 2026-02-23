#include <vector>
#include <memory>

#include "nn.h"
#include "Dataset.h"


class NNFramework {
    std::vector<std::unique_ptr<mynnLayer>> layers_;
    std::unique_ptr<mynnLossFunction> loss_;
    std::unique_ptr<mynnOptimizer> optimizer_;

    ComputationGraph graph_;
    bool updateWeights_;
    NNTaskType type_;

    bool training_mode_;

    public:
    NNFramework(NNTaskType type) : type_(type), training_mode_(true) {};
    void add(std::unique_ptr<mynnLayer> layer);
    void setLoss(std::unique_ptr<mynnLossFunction> layer);
    void setOptimizer(std::unique_ptr<mynnOptimizer> layer);
    void enableTrainingMode(bool training_mode) {training_mode_ = training_mode;};
    mynn::Mat forward(const mynn::Mat& input);
    backwardResult backward(const mynn::Mat& dl_dy);
    // void fit(inputData input, groundtruth output);
    void fit(const mynn::Mat& input, const mynn::Mat& output, int epoch, bool validation = false, const mynn::Mat& valinput = mynn::Mat(), const mynn::Mat& valoutput = mynn::Mat());
    void fit(const mynn::Dataset& train, int epoch, bool validation = false, const mynn::Dataset& test = mynn::Dataset());
    void eval(const mynn::Mat& input, const mynn::Mat& output);
    // void eval(const mynn::Mat& input, const mynn::Mat& output);

    double findAccuracyClassification(mynn::Mat y, mynn::Mat trueValues);

};