#pragma once


#include "Mat.h"
// #include <vector>
// #include <utility>
#include <string>

namespace mynn {
    class Dataset {
        public:
        mynn::Mat input_;
        mynn::Mat output_;
        void readCSVData(std::string filepath, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& outputs);
        
        //fasion
        static int32_t readBigEndianInt(std::ifstream& f);
        std::vector<double> loadMNISTImagesAsDouble(const std::string& path,
                                            int& num_images,
                                            int& rows,
                                            int& cols);
        std::vector<double> loadMNISTLabelsAsDouble(const std::string& path, int& num_labels);
        // fasion

        // public:
        Dataset();
        Dataset(std::string filepath);
        Dataset(std::string trainingfilepath, std::string testingfilepath);
        Dataset(Mat&& input, Mat&& output) : 
            input_(input), output_(output) {};

        Dataset(const Mat& input, const Mat& output) : 
            input_(input), output_(output) {};
        
        size_t size();
        void shuffle();
        std::pair<Dataset, Dataset> split(double trainratio);
    };

    template <typename T>
    void readNumbers(std::stringstream& ss, std::vector<T>& outVector, size_t count) {
        T value;
        size_t i = 0;
        std::string token;
        while (i < count && std::getline(ss, token, ',')) {
            std::stringstream tokenstream(token);
            if(tokenstream >> value) {
                outVector.push_back(value);
                ++i;
            }
        }
    }
}