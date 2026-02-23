#include <nn.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>


#include <sstream>
#include <string>

#include "NNFramework.h"


// Structure to hold input features and RGB values for each data point
struct DataPoint {
    std::vector<double> features;  // Input features
    std::vector<int> rgb;          // RGB output (R, G, B)
};

// Function to read CSV data from a file
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


static int32_t readBigEndianInt(std::ifstream& f) {
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

std::vector<double> loadMNISTImagesAsDouble(const std::string& path,
                                            int& num_images,
                                            int& rows,
                                            int& cols)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + path);

    int32_t magic = readBigEndianInt(file);
    num_images   = readBigEndianInt(file);
    rows         = readBigEndianInt(file);
    cols         = readBigEndianInt(file);

    if (magic != 2051)
        throw std::runtime_error("Invalid MNIST image file (wrong magic number)");

    size_t total_pixels = static_cast<size_t>(num_images) * rows * cols;
    std::vector<uint8_t> buffer(total_pixels);
    file.read(reinterpret_cast<char*>(buffer.data()), total_pixels);

    if (!file)
        throw std::runtime_error("File ended unexpectedly while reading image data");

    // Convert to normalized doubles in [0,1]
    std::vector<double> data(total_pixels);
    for (size_t i = 0; i < total_pixels; ++i)
        data[i] = static_cast<double>(buffer[i]) / 255.0;

    return data;
}
std::vector<double> loadMNISTLabelsAsDouble(const std::string& path, int& num_labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + path);

    int32_t magic = readBigEndianInt(file);
    num_labels    = readBigEndianInt(file);

    if (magic != 2049)
        throw std::runtime_error("Invalid MNIST label file (wrong magic number)");

    std::vector<uint8_t> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), num_labels);

    if (!file)
        throw std::runtime_error("File ended unexpectedly while reading labels");

    // Convert to double
    std::vector<double> labels(num_labels);
    for (int i = 0; i < num_labels; ++i)
        labels[i] = static_cast<double>(buffer[i]);

    return labels;
}

void printMNISTImage(const std::vector<double>& images, int index, int rows, int cols) {
    const int offset = index * rows * cols;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double val = images[offset + r * cols + c];

            // Pick a symbol depending on brightness
            char pixel;
            if (val > 0.75) pixel = '#';
            else if (val > 0.5) pixel = 'O';
            else if (val > 0.25) pixel = '.';
            else pixel = ' ';

            std::cout << pixel;
        }
        std::cout << "\n";
    }
}

void printMNISTImageFromMat(mynn::Mat& images, int index) {
    int inputsize = images.size().cols;
    int r = 28, c = 28;
    for(int i = 0; i < r; ++i) {
        for(int j=0; j < c;++j) {
            auto val = images(index, i*r + j);
            char pixel;

            if (val > 0.75) pixel = '#';
            else if (val > 0.5) pixel = 'O';
            else if (val > 0.25) pixel = '.';
            else pixel = ' ';

            std::cout << pixel;

        }
        std::cout << "\n";
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

int main2() {
    // get data

    int n, r, c;
    auto images = loadMNISTImagesAsDouble("../../examples/data/t10k-images-idx3-ubyte", n, r, c);
    // auto images = loadMNISTImagesAsDouble("../../examples/data/train-images-idx3-ubyte", n, r, c);
    int i = 10;
    // std::vector<double> ithimage(images.begin() + i * r * c, 
    //                             images.begin() + (i+1) * r * c);

    int nl;
    auto lables = loadMNISTLabelsAsDouble("../../examples/data/t10k-labels-idx1-ubyte", nl);
    // auto lables = loadMNISTLabelsAsDouble("../../examples/data/train-labels-idx1-ubyte", nl);
    mynn::Mat input(n, r*c, images);
    mynn::Mat groundTruth(lables);
    groundTruth = groundTruth.transpose();
    std::cout <<"no of images = "<<n<<", no of lables = "<<nl<<"\n";

    NNFramework mynn {NNTaskType::CLASSIFICATION};

    mynn.add(std::make_unique<DenseLayer>(784, 100));
    mynn.add(std::make_unique<ReLU>());
    mynn.add(std::make_unique<DropoutLayer>(0.1));

    mynn.add(std::make_unique<DenseLayer>(100, 10));

    mynn.setLoss(std::make_unique<A_Softmax_L_CatergoricalCrossEntropy>());
    mynn.setOptimizer(std::make_unique<OptimizerADAM>(0.02, 5e-7, true, 1e-7, 0.9, 0.999));

    mynn.fit(input, groundTruth, 1000);
    // mynn.fit(input, groundTruth, 2);

    mynn.eval(input, groundTruth);


    return 0;
}

int main() {
    // get data

    auto data = ConvertCSVtoMat();
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

int main22() {
    // get data

    auto data = ConvertCSVtoMat();
    mynn::Mat& input = data.first;
    mynn::Mat& groundTruth = data.second;
    // // return 0;
    // int n, r, c;
    // // auto images = loadMNISTImagesAsDouble("../../examples/data/t10k-images-idx3-ubyte", n, r, c);
    // auto images = loadMNISTImagesAsDouble("../../examples/data/train-images-idx3-ubyte", n, r, c);
    // int i = 10;
    // // std::vector<double> ithimage(images.begin() + i * r * c, 
    // //                             images.begin() + (i+1) * r * c);

    // int nl;
    // // auto lables = loadMNISTLabelsAsDouble("../../examples/data/t10k-labels-idx1-ubyte", nl);
    // auto lables = loadMNISTLabelsAsDouble("../../examples/data/train-labels-idx1-ubyte", nl);
    // mynn::Mat input(n, r*c, images);
    // mynn::Mat groundTruth(lables);
    // groundTruth = groundTruth.transpose();
    // std::cout <<"no of images = "<<n<<", no of lables = "<<nl;


    // printMNISTImage(images, i, r, c);
    // printMNISTImageFromMat(input, i);
    // std::cout <<"label = "<<lables[i]<<"\n";
    // std::cout <<"label = "<<groundTruth(i, 0)<<"\n";

    // architecture
    DenseLayer L1 {3, 128};
    ReLU activation1;
    DropoutLayer D1 {0.1};
    DenseLayer L2 {128, 64};
    ReLU activation2;
    DropoutLayer D2 {0.1};
    DenseLayer L3 {64, 32};
    ReLU activation3;
    DropoutLayer D3 {0.1};
    DenseLayer L4 {32, 3};
    MSE lossactivation;


    OptimizerADAM optimizer(0.02, 5e-7, true, 1e-7, 0.9, 0.999);



    // training
    int count = 30;
    count = 10000;
    for(int i = 0 ; i < count; ++i) {
        // forward
        auto L1output = L1.forward(input);
        auto activation1output = activation1.run(L1output);
        auto dropout1output = D1.forward(activation1output);
        auto L2output = L2.forward(dropout1output);
        auto activation2output = activation2.run(L2output);
        auto dropout2output = D2.forward(activation2output);
        
        auto L3output = L3.forward(dropout2output);
        auto activation3output = activation3.run(L3output);
        auto dropout3output = D3.forward(activation3output);

        auto L4output = L4.forward(dropout3output);
        auto output = lossactivation.forwardL(L4output, groundTruth);

        // if (i % 100 == 0)
        std::cout<<"epoch = "<<i<<", loss="<<output.meanof1d()<<"\n";

        // backward
        auto L4__dl_dz = lossactivation.backwardL(L4output, groundTruth);
        auto a = L4.backward(L4__dl_dz, dropout3output);

        auto b = D3.backward(std::get<2>(a));
        auto c = activation3.backward(b, L3output);
        auto d = L3.backward(c, dropout2output);

        auto e = D2.backward(std::get<2>(d));
        auto f = activation2.backward(e, L2output);
        auto g = L2.backward(f, dropout1output);

        auto h = D1.backward(std::get<2>(g));
        auto j = activation1.backward(h, L1output);
        auto k = L1.backward(j, input);

        
        // optimization
        optimizer.preUpdateParams();
        optimizer.updateParams(
            L1,
            std::get<0>(k),
            std::get<1>(k)
        );
        optimizer.updateParams(
            L2,
            std::get<0>(g),
            std::get<1>(g)
        );
        optimizer.updateParams(
            L3,
            std::get<0>(d),
            std::get<1>(d)
        );
        optimizer.updateParams(
            L4,
            std::get<0>(a),
            std::get<1>(a)
        );
        optimizer.postUpdateParams();
    }

    // testing
    return 0;
}