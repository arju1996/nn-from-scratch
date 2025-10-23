// #include <iostream>
#include <nn.h>

// #include <fstream>
// #include "print.h"
#include "random.h"

std::pair< std::vector<std::vector<double>>, std::vector<double>> spiltToInputAndOutputs(std::vector<std::tuple<double, double, int>> data) {
    std::vector<std::vector<double>> inputs;
    std::vector<double> outputs;
    for(auto item: data) {
        inputs.push_back(
            {
            std::get<0>(item),
            std::get<1>(item)
            }
        );
        outputs.push_back(std::get<2>(item));
    }
    return {inputs, outputs};
}

void spiltToInputAndOutputs(std::vector<std::tuple<double, double, int>> data, std::vector<std::vector<double>>& inputs, std::vector<double>& outputs) {
    for (auto item : data) {
        inputs.push_back(
            {
            std::get<0>(item),
            std::get<1>(item)
            }
        );
        outputs.push_back(std::get<2>(item));
    }
}

int findAccuracy(mynn::Mat y, mynn::Mat trueValues) {
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
    return correctCount*100/trueValues.size().rows;
}


int main() {
    std::vector<std::tuple<double, double, int>> data = math::dataset::GenerateSpiralData(3, 100, 0.2);
    std::pair< std::vector<std::vector<double>>, std::vector<double>> inputandoutput = 
            spiltToInputAndOutputs(data);

    auto inputbatch = inputandoutput.first;
    auto trueValues = inputandoutput.second;

    mynn::Mat input(inputbatch);
    mynn::Mat groundTruth(trueValues);

    groundTruth = groundTruth.transpose();

    //return 0;


    // network
    DenseLayer L1 {2, 64, 0, 0, 5e-4, 5e-4};
    ReLU activation1;
    DropoutLayer D1 {0.1};
    DenseLayer L2 {64, 3};
    A_Softmax_L_CatergoricalCrossEntropy lossactivation;
    OptimizerADAM optimizer(0.02, 5e-7, true, 1e-7, 0.9, 0.999);

    // training
    int count = 10000;

    for(int i = 0 ; i < count; ++i) {
        // forward
        auto L1output = L1.forward(input);
        auto activation1output = activation1.run(L1output);
        auto dropout1output = D1.forward(activation1output);
        auto L2output = L2.forward(dropout1output);
        auto lossactivationoutput__lose = lossactivation.forward(L2output, groundTruth);

        if (i % 100 == 0) {
            mynn::Mat lossCol = std::get<1>(lossactivationoutput__lose);
            double acc = findAccuracy(std::get<0>(lossactivationoutput__lose), groundTruth);
            double reguloss = lossactivation.regularizationLoss(&L1) + lossactivation.regularizationLoss(&L2);
            std::cout<<"lose = "<<lossCol.meanof1d()<<"\t";
            std::cout<<"reg loss = "<<reguloss<<" total loss = "<<(lossCol.meanof1d() + reguloss)<<", accu = "<<acc;
            std::cout<<" lr = "<<optimizer.getCurrentLearningRate()<<"\n";
        }

        // backward
        auto L2__dl_dz = lossactivation.backward(std::get<0>(lossactivationoutput__lose), groundTruth);
        auto L2__tupl_dl_dw___dl_db___dl_dx = L2.backward(L2__dl_dz, activation1output);
        auto l2_dl_dx = D1.backward(std::get<2>(L2__tupl_dl_dw___dl_db___dl_dx));
        auto L1__dl_dz = activation1.backward(l2_dl_dx, L1output);
        auto L1__tupl_dl_dw___dl_db___dl_dx = L1.backward(L1__dl_dz, input);

        
        // optimization
        optimizer.preUpdateParams();
        optimizer.updateParams(
            L1,
            std::get<0>(L1__tupl_dl_dw___dl_db___dl_dx),
            std::get<1>(L1__tupl_dl_dw___dl_db___dl_dx)
        );
        optimizer.updateParams(
            L2,
            std::get<0>(L2__tupl_dl_dw___dl_db___dl_dx),
            std::get<1>(L2__tupl_dl_dw___dl_db___dl_dx)
        );
        optimizer.postUpdateParams();
    }


    std::cout<<"testing"<<"\n";
    auto testindata = math::dataset::GenerateTestingSpiralData(3, 100, 0.2);
    std::pair< std::vector<std::vector<double>>, std::vector<double>> testinginputandoutput = 
            spiltToInputAndOutputs(testindata);
    auto testinginputbatch = testinginputandoutput.first;
    auto testingtrueValues = testinginputandoutput.second;
    mynn::Mat testinginput(testinginputbatch);
    mynn::Mat testinggroundTruth(testingtrueValues);

    testinggroundTruth = testinggroundTruth.transpose();

    auto testL1output = L1.forward(testinginput);
    auto testactivation1output = activation1.run(testL1output);
    auto dropout1output = D1.forward(testactivation1output);
    auto testL2output = L2.forward(dropout1output);
    auto testlossactivationoutput__lose = lossactivation.forward(testL2output, testinggroundTruth);

    mynn::Mat testlossCol = std::get<1>(testlossactivationoutput__lose);
    double testacc = findAccuracy(std::get<0>(testlossactivationoutput__lose), testinggroundTruth);
    std::cout<<"lose = "<<testlossCol.meanof1d()<<"\t"<<", accu = "<<testacc;
    std::cout<<" lr = "<<optimizer.getCurrentLearningRate()<<"\n";
}