#include <iostream>
#include <array>
#include <vector>

#include <cassert>

#include "random.h"

class DenseLayer {
    int input_size_;
    int no_of_nodes_;
    std::vector<std::vector<double>> weights_;
    std::vector<double> bias_;
public:
    DenseLayer(int input_size, int no_of_nodes);
    void PrintDenseLayer();
    void whatever();
    inline double VectorDotVector(std::vector<double> a, std::vector<double> b);
    std::vector<double> run(std::vector<double> input);
    std::vector<std::vector<double>> runInputBatch(std::vector<std::vector<double>> inputBatch);
};