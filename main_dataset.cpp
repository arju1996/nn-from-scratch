#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <sstream>
#include <string>


#include <nn.h>
#include "NNFramework.h"
#include "Dataset.h"


void readCSVData(const std::string& filename, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs) {
    std::ifstream file(filename);  // Open the CSV file
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Skip the header line
    std::getline(file, line);

    // Read each data line from the CSV
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        // Temporary variables to hold parsed features and RGB values
        std::vector<double> input_features;
        std::vector<double> rgb_values;

        // Read input features (3 features)
        for (int i = 0; i < 3; ++i) {
            std::getline(ss, value, ',');  // Split by comma
            input_features.push_back(std::stod(value));  // Convert to double and add to features
        }

        // Read RGB values (3 values)
        for (int i = 0; i < 3; ++i) {
            std::getline(ss, value, ',');
            rgb_values.push_back(std::stod(value));  // Convert to int and add to rgb
        }

        // Store parsed features and RGB values
        inputs.push_back(input_features);
        outputs.push_back(rgb_values);
    }

    // Close the file
    file.close();
}

std::pair<mynn::Mat, mynn::Mat> ConvertCSVtoMat() {
    std::string filename = "../../python/synthetic_rgb_dataset.csv";  // Path to your CSV file

    // Vectors to store input features and RGB outputs
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;

    // Read CSV data into inputs and outputs
    readCSVData(filename, inputs, outputs);

    // Create mynn::Mat objects for inputs and outputs
    int num_samples = inputs.size();
    int num_features = inputs[0].size();
    int num_outputs = outputs[0].size();

    std::cout<<"num_samples=" << num_samples<<", num_features="<<num_features<<", num_outputs="<<num_outputs<<"\n";

    mynn::Mat inputMat(inputs);  // Matrix to hold input data
    mynn::Mat outputMat(outputs);  // Matrix to hold output data
    std::cout<<"input shape = "<< inputMat.size().rows<<","<<inputMat.size().cols<<"\n";
    std::cout<<"output shape = "<< outputMat.size().rows<<","<<outputMat.size().cols<<"\n";


    // Print the input and output matrices (for debugging)
    // std::cout << "Input Matrix:\n";
    // inputMat.print();
    // std::cout << "\nOutput Matrix:\n";
    // outputMat.print();
    return {inputMat, outputMat};
}

int mainclassification() {
    
    std::string filenametrain = "../../examples/data/t10k-images-idx3-ubyte";  // Path to your CSV file
    std::string filenametest = "../../examples/data/t10k-labels-idx1-ubyte";  // Path to your CSV file
    mynn::Dataset wholedata(filenametrain, filenametest);
    wholedata.shuffle();
    auto trainandtest = wholedata.split(0.8);

    // data part

    NNFramework mynn {NNTaskType::CLASSIFICATION};

    mynn.add(std::make_unique<DenseLayer>(784, 100));
    mynn.add(std::make_unique<ReLU>());
    mynn.add(std::make_unique<DropoutLayer>(0.1));

    mynn.add(std::make_unique<DenseLayer>(100, 10));

    mynn.setLoss(std::make_unique<A_Softmax_L_CatergoricalCrossEntropy>());
    mynn.setOptimizer(std::make_unique<OptimizerADAM>(0.02, 5e-7, true, 1e-7, 0.9, 0.999));

    mynn.fit(trainandtest.first, 1000, true, trainandtest.second);
    return 0;
}

int mainregression() {
    std::string filename = "../../python/synthetic_rgb_dataset.csv";  // Path to your CSV file
    mynn::Dataset wholedata(filename);
    wholedata.shuffle();
    auto trainandtest = wholedata.split(0.8);
    //

    // mynn::Mat& input = data.first;
    // mynn::Mat& groundTruth = data.second;

    NNFramework mynn {NNTaskType::REGRESSION};

    mynn.add(std::make_unique<DenseLayer>(3, 128));
    mynn.add(std::make_unique<ReLU>());
    mynn.add(std::make_unique<DropoutLayer>(0.1));

    mynn.add(std::make_unique<DenseLayer>(128, 64));
    mynn.add(std::make_unique<ReLU>());
    mynn.add(std::make_unique<DropoutLayer>(0.1));

    mynn.add(std::make_unique<DenseLayer>(64, 32));
    mynn.add(std::make_unique<ReLU>());
    mynn.add(std::make_unique<DropoutLayer>(0.1));

    mynn.add(std::make_unique<DenseLayer>(32, 3));

    mynn.setLoss(std::make_unique<MSE>());
    mynn.setOptimizer(std::make_unique<OptimizerADAM>(0.02, 5e-7, true, 1e-7, 0.9, 0.999));

    mynn.fit(trainandtest.first, 1000, true, trainandtest.second);

    return 0;

}

int main22() {
    // get data

    auto data = ConvertCSVtoMat();

    mynn::Dataset wholedata{data.first, data.second};
    wholedata.shuffle();
    std::cout<<"whole size = "<<wholedata.size()<<"\n";
    auto testandtrain = wholedata.split(0.8);
    std::cout<<"train size = "<<testandtrain.first.size()<<"\n";
    std::cout<<"test size = "<<testandtrain.second.size()<<"\n";
    
    
    std::string filename = "../../python/synthetic_rgb_dataset.csv";  // Path to your CSV file
    mynn::Dataset d(filename);
    d.shuffle();
    std::cout<<"2whole size = "<<d.size()<<"\n";
    auto testandtrain2 = d.split(0.8);
    std::cout<<"2train size = "<<testandtrain2.first.size()<<"\n";
    std::cout<<"2test size = "<<testandtrain2.second.size()<<"\n";

    // return 0;

    mynn::Mat& input = data.first;
    mynn::Mat& groundTruth = data.second;

    NNFramework mynn {NNTaskType::REGRESSION};

    mynn.add(std::make_unique<DenseLayer>(3, 128));
    mynn.add(std::make_unique<ReLU>());
    mynn.add(std::make_unique<DropoutLayer>(0.1));

    mynn.add(std::make_unique<DenseLayer>(128, 64));
    mynn.add(std::make_unique<ReLU>());
    mynn.add(std::make_unique<DropoutLayer>(0.1));

    mynn.add(std::make_unique<DenseLayer>(64, 32));
    mynn.add(std::make_unique<ReLU>());
    mynn.add(std::make_unique<DropoutLayer>(0.1));

    mynn.add(std::make_unique<DenseLayer>(32, 3));

    mynn.setLoss(std::make_unique<MSE>());
    mynn.setOptimizer(std::make_unique<OptimizerADAM>(0.02, 5e-7, true, 1e-7, 0.9, 0.999));

    mynn.fit(input, groundTruth, 1000);

    return 0;
}


int main() {
    mainclassification();
}