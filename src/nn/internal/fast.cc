#include "fast.h"

mynn::Mat mynn::fast::dense_forward(const mynn::Mat& input, const mynn::Mat& weight, const mynn::Mat& bias) {
    /*
    mynn::Mat result = input.multiply(weights_.transpose()) + bias_;
    */

    mynn::Size inputSize = input.size();
    mynn::Size weightSize = weight.size();
    mynn::Size biasSize = bias.size();

    if(inputSize.cols != weightSize.cols) {
        throw std::invalid_argument("Cannot multiply.");
    }

    if(weightSize.rows != biasSize.cols) {
        throw std::invalid_argument("Mats are diff dim, cant add.");
    }

    Mat result(inputSize.rows, weightSize.rows);

    for(size_t i = 0, r = inputSize.rows; i < r; ++i) {
        for(size_t j=0, c = weightSize.rows; j < c ; ++j) {
            double pos = bias(0, j);
            for(size_t k=0, s = inputSize.cols; k < s; ++k) {
                pos += input(i, k) * weight(j, k);
            }
            result(i,j) = pos;
        }
    }
    return result;
}

mynn::Mat mynn::fast::relu_forward(const mynn::Mat& input) {
    const auto* idata = input.data();
    const int total = input.size().count();
    mynn::Mat result(input.size().rows, input.size().cols);
    auto* rdata = result.data();
    for(size_t i = 0; i < total; ++i) {
        rdata[i] = std::max(idata[i], 0.0);
    }
    return result;
}

mynn::Mat mynn::fast::drop_forward(const mynn::Mat& input, const mynn::Mat& mask) {
    const auto* idata = input.data();
    const auto* mdata = mask.data();
    const int total = input.size().count();
    mynn::Mat result(input.size().rows, input.size().cols);
    auto* rdata = result.data();
    for(size_t i = 0; i < total; ++i) {
        rdata[i] = idata[i] * mdata[i];
    }
    return result;
}

mynn::Mat mynn::fast::softmax_forward(const mynn::Mat& input) {
    const int cols = input.size().cols;
    const int rows = input.size().rows;
    mynn::Mat output(input.size().rows, input.size().cols);
    const auto* idata = input.data();
    auto* odata = output.data();
    for (int i = 0; i<rows; ++i) {
        auto inputith_row = idata + i*cols;
        auto outputith_row = odata + i*cols;

        // find max;
        double max = inputith_row[0];
        for(int j = 0; j < cols; ++j) {
            if (inputith_row[j] > max)
                max = inputith_row[j];
        }

        // find esp and sums
        double sumofexp = 0;
        for(int j = 0; j < cols; ++i) {
            double exp = std::exp(inputith_row[j] - max);
            outputith_row[j] = exp;
            sumofexp += exp;
        }

        // div with sumexp
        const double inv = 1.0 / sumofexp;
        for(int j = 0; j < cols; ++j) {
            outputith_row[j] *= inv;
        }
    }
    return output;
}