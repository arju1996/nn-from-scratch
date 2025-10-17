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

int inferenceAndAccuracy() {
    int inputsize = 4;
    int nodecount = 3;
    /*
    create a Layer,
    1. give no of inputs, no of nodes
    2. initialize random weights and biases;
    */
    DenseLayer L1 {inputsize, nodecount};
    L1.PrintDenseLayer();
    ReLU activation1;




    std::vector<double> inpu = {1, 2, 3, 4};
    mynn::Mat input = mynn::Mat(1, 4, inpu);
    // print::PrintVectorWithLabel(input, "\ninput");
    std::cout<<"input = \n"<<input<<"\n";
    mynn::Mat result1 = L1.forward(input);
    std::cout<<"result1 = \n"<<result1<<"\n";
    auto result2 = activation1.run(result1);
    std::cout<<"result2 = \n"<<result1<<"\n";


    std::vector<std::vector<double>> inputbatch = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    mynn::Mat input2 = mynn::Mat(inputbatch);

    std::cout<<"input = \n"<<input2<<"\n";
    auto result1forbatch = L1.forward(input2);
    std::cout<<"result1 batch \n= "<<result1forbatch<<"\n";
    auto result2forbatch = activation1.run(result1forbatch);
    std::cout<<"result2 batch \n= "<<result2forbatch<<"\n";

    
    return 0;
}


//

int trainingForSingleLayerOneBatch() {
    int inputsize = 4;
    int nodecount = 3;

    DenseLayer L1 {inputsize, nodecount};
    L1.setWeights(mynn::Mat({{0.1, 0.2, 0.3, 0.4} , {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}}));
    L1.setBias(mynn::Mat({0.1, 0.2, 0.3}));
    ReLU activation1;

    std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
    auto result = L1.forward(mynn::Mat(input));
    // std::vector<double> resultafteractivation = activation1.run(result);

    print::PrintVectorWithLabel(input, "\ninput");
    // print::PrintVectorWithLabel(result, "\nresult");
    std::cout<<"result1 \n= "<<result<<"\n";
    // print::PrintVectorWithLabel(resultafteractivation, "\nresultafteractivation");

        // double sum = 0;
        // for  (auto item: resultafteractivation)
        // sum += item;
        // std::cout <<" loss = "<<sum * sum <<"\n";

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
        // print::PrintVectorWithLabel(weight, "\nweight");
        std::cout<<"weight = \n" << weight;
        
        // dl/dw is a matrix of 4 * 3;
        // weight is of 3 * 4
        // w11 = w11 - learning_rate * dl_dw11;
        for(int i = 0; i<inputsize; ++i) {
            for (int j = 0; j < nodecount; ++j) {
                weight(j, i) = weight(j, i) - learning_rate * dl_dw[i][j];
            }
        }
        L1.setWeights(weight);

        auto bias = L1.getBias();
        for (int j = 0; j < nodecount; ++j) {
            bias(0, j) = bias(0, j) - learning_rate * dl_dz[j];
        }
        L1.setBias(bias);


        std::vector<double> input = {1.0, 2.0, 3.0, 4.0};
        auto result = L1.forward(mynn::Mat(input));
        // std::vector<double> resultafteractivation = activation1.run(result);
        // double sum = 0;
        // for  (auto item: resultafteractivation)
        // sum += item;
        // std::cout <<" loss = "<<sum * sum << " after "<<iter<<"th iteration\n";
    }

    return 0;
}

void chapter19() {
    // int inputsize = 4;
    // int nodecount = 3;
    /*
    create a Layer,
    1. give no of inputs, no of nodes
    2. initialize random weights and biases;
    */
    // DenseLayer L1 {inputsize, nodecount};
    // L1.PrintDenseLayer();
    ReLU activation1;

    A_Softmax_L_CatergoricalCrossEntropy S_CCE;
    mynn::Mat softmax_output({{0.7, 0.1, 0.2} , {0.1, 0.5, 0.4}, {0.02, 0.9, 0.08}});
    mynn::Mat class_targets({0, 1, 1});
    auto loss = S_CCE.backward(softmax_output, class_targets);

    std::cout<<"final result = "<<loss<<"\n";
}

void chapter21fullnn() {
    std::vector<std::tuple<double, double, int>> data = math::dataset::GenerateSineDataV(3, 2, 0.2);
    std::pair< std::vector<std::vector<double>>, std::vector<double>> inputandoutput = 
            spiltToInputAndOutputs(data);
    auto inputbatch = inputandoutput.first;
    auto trueValues = inputandoutput.second;
    print::PrintVectorWithLabel(inputbatch, "\ninput");
    print::PrintVectorWithLabel(trueValues, "\ntrueValues");

    mynn::Mat input(inputbatch);
    mynn::Mat groundTruth(trueValues);

    groundTruth = groundTruth.transpose();


    
    // create the blocks
    int L1inputsize = 2;
    int L1nodecount = 3;

    int L2inputsize = 3;
    int L2nodecount = 3;

    DenseLayer L1 {L1inputsize, L1nodecount};
    ReLU activation1;
    DenseLayer L2 {L2inputsize, L2nodecount};
    A_Softmax_L_CatergoricalCrossEntropy lossactivation;

    L1.PrintDenseLayer();
    L2.PrintDenseLayer();

    // forward
    auto L1output = L1.forward(input);
    std::cout<<"L1output = "<<L1output<<"\n";

    auto activation1output = activation1.run(L1output);
    std::cout<<"activation1output = "<<activation1output<<"\n";

    auto L2output = L2.forward(activation1output);
    auto lossactivationoutput__lose = lossactivation.forward(L2output, groundTruth);

    std::cout<<"lossactivationoutput = "<<std::get<0>(lossactivationoutput__lose)<<"\n";
    std::cout<<"lose = "<<std::get<1>(lossactivationoutput__lose)<<"\n";

    
    // backward
    auto L2__dl_dz = lossactivation.backward(std::get<0>(lossactivationoutput__lose), groundTruth);
    auto L2__tupl_dl_dw___dl_db___dl_dx = L2.backward(L2__dl_dz, activation1output);
    auto L1__dl_dz = activation1.backward(std::get<2>(L2__tupl_dl_dw___dl_db___dl_dx), L1output);
    auto L1__tupl_dl_dw___dl_db___dl_dx = L1.backward(L1__dl_dz, input);


    std::cout<<"L1 weights deri = "<<std::get<0>(L1__tupl_dl_dw___dl_db___dl_dx)<<"\n";
    std::cout<<"L1 bias deri = "<<std::get<1>(L1__tupl_dl_dw___dl_db___dl_dx)<<"\n";
    std::cout<<"L1 input deri = "<<std::get<2>(L1__tupl_dl_dw___dl_db___dl_dx)<<"\n";

    std::cout<<"L2 weights deri = "<<std::get<0>(L2__tupl_dl_dw___dl_db___dl_dx)<<"\n";
    std::cout<<"L2 bias deri = "<<std::get<1>(L2__tupl_dl_dw___dl_db___dl_dx)<<"\n";
    std::cout<<"L2 input deri = "<<std::get<2>(L2__tupl_dl_dw___dl_db___dl_dx)<<"\n";

}

// int findAccuracy(std::vector<std::vector<double>> y, std::vector<int> trueValues) {
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
    // inferenceAndAccuracy();
    // trainingForSingleLayerOneBatch();
    // trainingForSingleLaterMultipleBatch();
    // findGradientOfInput();

    // chapter19();
    chapter21fullnn();



}