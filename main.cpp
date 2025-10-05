#include <array>
#include <cassert>
#include <iostream>
#include <vector>

#include "print.h"
#include "random.h"

#include <nn.h>

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

int inferenceAndAccuracy() {
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


int trainingForSingleLayerOneBatch() {
    int inputsize = 4;
    int nodecount = 3;

    DenseLayer L1 {inputsize, nodecount};
    L1.setWeights({{0.1, 0.2, 0.3, 0.4} , {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}});
    L1.setBias({0.1, 0.2, 0.3});
    ReLU activation1;

    std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> result = L1.run(input);
    std::vector<double> resultafteractivation = activation1.run(result);

    print::PrintVectorWithLabel(input, "\ninput");
    print::PrintVectorWithLabel(result, "\nresult");
    print::PrintVectorWithLabel(resultafteractivation, "\nresultafteractivation");

        double sum = 0;
        for  (auto item: resultafteractivation)
        sum += item;
        std::cout <<" loss = "<<sum * sum <<"\n";

    /*
    connections
    L = (y)^2; y = a1 + a2 + a3;
    a1 = relu(z1)
    z1 = w11x1 + w12x2+ w13x3 + w13x3 + b1;
    */

    /*
    values
    z1 = 3, z2 = 7.2, z3 = 11.4
    L = 21.6^2;
    */

    /*
    variables
    dL_dw11 = dL_dy * dy_drelu() * drelu()_dz1 * dz1_dw11
    dl_dy = 2y;
    dy_drelu = 1;
    drelu_dz1 = 1 when z1 is postive, 0 when z1 is negative
    so dl_dz1 = 2 * 21.6 * 1 * 1 = 43.2?
    dz1_dw11 = x1;

    dl_dw11 = dl_dz1 * dz1_dw11;
    **** dl/dw =  Xt * dl_dz ****

    dL_db1 = dL_dy * dy_drelu() * drelu()_dz1 
    dl_db1 =  dl_dz
    **** dl_db =  dl_dz ****
    */

    int iterations = 180;
    iterations = 1;
    double learning_rate = 0.001;

    for(int iter = 0; iter < iterations; ++iter) {
        std::vector<double> z = L1.run(input);
        std::vector<double> relu;
        for(auto item: z) {
            if (item > 0)
                relu.push_back(item);
            else
                relu.push_back(0);
        }
        double y = 0;
        for (auto item: relu)
        y += item;

        std::vector<double> relu_deri;
        for(auto item: relu) {
            relu_deri.push_back(item>0);
        }

        std::vector<double> dl_dz;
        for(auto item: relu_deri) {
            dl_dz.push_back(2 * y * item);
        }

        print::PrintVectorWithLabel(dl_dz, "\ndl_dz");

        std::vector<double> xt = input;
        // dl_dw = xt * dl_dz;
        std::vector<std::vector<double>> dl_dw (inputsize, std::vector<double>(nodecount));
        for(int i = 0; i<inputsize; ++i) {
            for(int j = 0; j<nodecount; ++j) {
                dl_dw[i][j] = xt[i] * dl_dz[j];
            }
        }

        print::PrintVectorWithLabel(dl_dw, "\ndl_dw");

        
        auto weight = L1.getWeights();
        print::PrintVectorWithLabel(weight, "\nweight");
        
        // dl/dw is a matrix of 4 * 3;
        // weight is of 3 * 4
        // w11 = w11 - learning_rate * dl_dw11;
        for(int i = 0; i<inputsize; ++i) {
            for (int j = 0; j < nodecount; ++j) {
                weight[j][i] = weight[j][i] - learning_rate * dl_dw[i][j];
            }
        }
        L1.setWeights(weight);

        auto bias = L1.getBias();
        for (int j = 0; j < nodecount; ++j) {
            bias[j] = bias[j] - learning_rate * dl_dz[j];
        }
        L1.setBias(bias);


        std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
        std::vector<double> result = L1.run(input);
        std::vector<double> resultafteractivation = activation1.run(result);
        double sum = 0;
        for  (auto item: resultafteractivation)
        sum += item;
        std::cout <<" loss = "<<sum * sum << " after "<<iter<<"th iteration\n";
    }

    return 0;
}


int main() {
    // inferenceAndAccuracy();
    // trainingForSingleLayerOneBatch();
}