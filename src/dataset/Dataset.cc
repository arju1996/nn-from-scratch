#include "Dataset.h"

#include <numeric>
#include <algorithm>
#include <random>


#include <fstream>
#include <sstream>
#include <filesystem>

mynn::Dataset::Dataset(std::string filepath) {
    mynn::Mat inputMat;
    mynn::Mat outputMat;

    std::filesystem::path path(filepath);
    if(path.extension() == ".csv") {
        // the structure of the data is encoded in the parsing logic itself;
        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> outputs;
        readCSVData(filepath, inputs, outputs);
        inputMat = mynn::Mat(inputs);
        outputMat = mynn::Mat(outputs);
    } else {
        std::cout<<"aa path.filename() = "<<path.filename();
    }


    input_ = std::move(inputMat);
    output_ = std::move(outputMat);
}


mynn::Dataset::Dataset(std::string trainingfilepath, std::string testingfilepath) {
    mynn::Mat inputMat;
    mynn::Mat outputMat;

    std::filesystem::path trainpath(trainingfilepath);
    std::filesystem::path testpath(testingfilepath);

    if(trainpath.extension() == ".csv" && testpath.extension() == ".csv") {
        // testing and training is in seperate csvs
    } else if (
        trainpath.filename().string() == "t10k-images-idx3-ubyte"
        &&
        testpath.filename().string() == "t10k-labels-idx1-ubyte"
    ) {
        // training and testing is in seperate fasion files
        int n, r, c;
        auto images = loadMNISTImagesAsDouble(trainpath.string(), n, r, c);
        inputMat = mynn::Mat(n, r*c, images);

        int nl;
        auto lables = loadMNISTLabelsAsDouble(testpath.string(), nl);
        outputMat = mynn::Mat(lables);
        outputMat = outputMat.transpose();
    } else {
        // something else
    }

    input_ = std::move(inputMat);
    output_ = std::move(outputMat);
}



// fasion
int32_t mynn::Dataset::readBigEndianInt(std::ifstream& f) {
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

std::vector<double> mynn::Dataset::loadMNISTImagesAsDouble(const std::string& path,
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

std::vector<double> mynn::Dataset::loadMNISTLabelsAsDouble(const std::string& path, int& num_labels) {
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

// fasion

void mynn::Dataset::readCSVData(std::string filepath, std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output) {
    std::ifstream file(filepath);
    std::string line;

    if(!file.is_open()) {
        std::cerr << "couldnt open file! \n";
        return;
    }
    std::getline(file, line);    

    // std::vector<std::vector<double>> input;
    // std::vector<std::vector<double>> output; //
    while(std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        
        // parsing logic
        std::vector<double> features;
        std::vector<double> outputs; //
        readNumbers(ss, features, 3);
        readNumbers(ss, outputs, 3);

        input.push_back(features);
        output.push_back(outputs);
    }

}



size_t mynn::Dataset::size() {
    // assumes each data is stored in each row
    return input_.size().rows;
}

void mynn::Dataset::shuffle() {
    std::vector<size_t> indices(size());
    std::iota(indices.begin(), indices.end(), 0);
    unsigned int seed = 234;
    // std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    std::shuffle(indices.begin(), indices.end(), std::mt19937{seed});

    
    mynn::Mat shuffled_inputs(input_.size().rows, input_.size().cols);
    mynn::Mat shuffled_outputs(output_.size().rows, output_.size().cols);
    for(size_t i = 0; i < size(); ++i) {
        shuffled_inputs.setRow(i, input_.row(indices[i]));
        shuffled_outputs.setRow(i, output_.row(indices[i]));
    }
    input_ = std::move(shuffled_inputs);
    output_ = std::move(shuffled_outputs);
}

std::pair<mynn::Dataset, mynn::Dataset>  mynn::Dataset::split(double trainratio) {
    size_t trainsize = static_cast<size_t> (trainratio * size());
    auto inputtrainingandtest = input_.splitRows(trainsize);
    auto outputtrainingandtest = output_.splitRows(trainsize);
    return {
        Dataset(std::move(inputtrainingandtest.first), std::move(outputtrainingandtest.first)), 
        Dataset(std::move(inputtrainingandtest.second), std::move(outputtrainingandtest.second))
    };
}
