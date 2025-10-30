#include <nn.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

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

int main() {
    // get data
    int n, r, c;
    // auto images = loadMNISTImagesAsDouble("../../examples/data/t10k-images-idx3-ubyte", n, r, c);
    auto images = loadMNISTImagesAsDouble("../../examples/data/train-images-idx3-ubyte", n, r, c);
    int i = 10;
    // std::vector<double> ithimage(images.begin() + i * r * c, 
    //                             images.begin() + (i+1) * r * c);

    int nl;
    // auto lables = loadMNISTLabelsAsDouble("../../examples/data/t10k-labels-idx1-ubyte", nl);
    auto lables = loadMNISTLabelsAsDouble("../../examples/data/train-labels-idx1-ubyte", nl);
    mynn::Mat input(n, r*c, images);
    mynn::Mat groundTruth(lables);
    groundTruth = groundTruth.transpose();
    std::cout <<"no of images = "<<n<<", no of lables = "<<nl;


    printMNISTImage(images, i, r, c);
    printMNISTImageFromMat(input, i);
    std::cout <<"label = "<<lables[i]<<"\n";
    std::cout <<"label = "<<groundTruth(i, 0)<<"\n";

    // architecture
    DenseLayer L1 {784, 100};
    ReLU activation1;
    DropoutLayer D1 {0.1};
    DenseLayer L2 {100, 10};
    A_Softmax_L_CatergoricalCrossEntropy lossactivation;
    OptimizerADAM optimizer(0.02, 5e-7, true, 1e-7, 0.9, 0.999);



    // training
    int count = 30;
    // count = 1;
    for(int i = 0 ; i < count; ++i) {
        // forward
        auto L1output = L1.forward(input);
        auto activation1output = activation1.run(L1output);
        auto dropout1output = D1.forward(activation1output);
        auto L2output = L2.forward(dropout1output);
        auto lossactivationoutput__lose = lossactivation.forward(L2output, groundTruth);

        if (i % 100 == 0 || true) {
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

    // testing
}