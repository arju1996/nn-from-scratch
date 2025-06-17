#include <vector>
#include <cassert>

// #include "random.h"
// ideally i could create an activation class instead
// and pass type of activation to constructor
// as the idea is same just the processing is different

class ReLU {
public:
    ReLU();
    std::vector<double> run(std::vector<double> input);
    std::vector<std::vector<double>> runInputBatch(std::vector<std::vector<double>> inputBatch);
};