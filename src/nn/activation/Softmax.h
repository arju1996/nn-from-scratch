#include <vector>
#include <cassert>

class Softmax {
public:
    std::vector<double> run(std::vector<double> input);
    std::vector<std::vector<double>> runInputBatch(std::vector<std::vector<double>> inputBatch);
};