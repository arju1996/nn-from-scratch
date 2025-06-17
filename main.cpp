#include <iostream>
#include <array>
#include <vector>

#include <cassert>
#include <DenseLayer.h>
#include <ReLU.h>
#include <Softmax.h>

#include "print.h"
#include "random.h"
#include "LossFunction.h"

int findAccuracy(std::vector<std::vector<double>> y, std::vector<int> trueValues) {
    int correctCount = 0;
    for(int i = 0 ; i < y.size(); ++i) {
        auto prediction = y.at(i);
        auto max = std::max_element(prediction.begin(), prediction.end());
        auto dist = std::distance(prediction.begin(), max);
        if(dist == trueValues.at(i))
            correctCount++;
    }

    std::cout <<"correctCount = "<<correctCount;
    return correctCount*100/trueValues.size();
}

std::pair< std::vector<std::vector<double>>, std::vector<int>> spiltToInputAndOutputs(std::vector<std::tuple<double, double, int>> data) {
    std::vector<std::vector<double>> inputs;
    std::vector<int> outputs;
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

int main() {
    int inputsize = 2;
    int nodecount = 2;
    /*
    create a Layer,
    1. give no of inputs, no of nodes
    2. initialize random weights and biases;
    */
    DenseLayer L1 {inputsize, nodecount};
    L1.PrintDenseLayer();
    ReLU activation1;
    DenseLayer L2 {2, 3};
    Softmax activation2;




    std::vector<double> input = {1, 2};
    print::PrintVectorWithLabel(input, "\ninput");
    std::vector<double> result1 = L1.run(input);
    print::PrintVectorWithLabel(result1, "\nresult1");
    std::vector<double> result1afteractivation = activation1.run(result1);
    print::PrintVectorWithLabel(result1afteractivation, "\nresult1afteractivation");

    std::vector<double> result2 = L2.run(result1afteractivation);
    print::PrintVectorWithLabel(result2, "\nresult2");
    std::vector<double> result2afteractivation = activation2.run(result2);
    print::PrintVectorWithLabel(result2afteractivation, "\nresult2afteractivation");
    
    /*
    std::vector<std::vector<double>> inputbatch = math::dataset::GenerateLinearDataV(2, 50, 0.2);
    */
    std::vector<std::tuple<double, double, int>> data = math::dataset::GenerateSineDataV(3, 50, 0.2);
    std::pair< std::vector<std::vector<double>>, std::vector<int>> inputandoutput = 
            spiltToInputAndOutputs(data);
    

    auto inputbatch = inputandoutput.first;

    print::PrintVectorWithLabel(inputbatch, "\n\n\n\ninputbatch");
    auto result1forbatch = L1.runInputBatch(inputbatch);
    auto result1forbatchafteractivation = activation1.runInputBatch(result1forbatch);

    auto result2forbatch = L2.runInputBatch(result1forbatchafteractivation);
    auto result2forbatchafteractivation = activation2.runInputBatch(result2forbatch);
    print::PrintVectorWithLabel(result2forbatchafteractivation, "\nresult2forbatchafteractivation");


    LossFunction CategoricalCrossEntropy;
    std::vector<int> trueValues;
    /*
    for(int i = 0 ; i<100; ++i)
        trueValues.push_back(0);
    */
    trueValues = inputandoutput.second;
    double loss = CategoricalCrossEntropy.calculate(result2forbatchafteractivation, trueValues);
    std::cout<<"\n\nfinal loss = "<<loss;

    int accu = findAccuracy(result2forbatchafteractivation, trueValues);
    std::cout<<"\nfinal accu = "<<accu;

    // auto data = math::dataset::GenerateSpiralData(3, 100);  // 3 classes, 100 points per class

    // for (const auto& p : data) {
    //     std::cout << p.x << ", " << p.y << " -> class " << p.label << '\n';
    // }
    return 0;
}