#include <vector>
#include <cassert>
#include <cmath>

// this usually is called on batch
class CategoricalCrossEntropy {
    int a;
public:
    int calculate(void* predicted_output, void* true_output);
    // i might need 


    // case when outputs are labels, ie not hot 1 encoded;
    // see if i can do xor to improve performance
    std::vector<double> forward(
        std::vector<std::vector<double>> resultafteractivationforabatch ,
        std::vector<int> classTargets
    );

    void clip(double &x, double lower, double upper);
    void clip(
        std::vector<std::vector<double>>& data,
        double lower,
        double upper
    );


    // case when outputs are hot 1 encoded
    void calculate(
        std::vector<std::vector<double>> resultafteractivationforabatch,
        std::vector<std::vector<int>> classTargets
    );
};

// ideally Lossfunction should be the parent
class LossFunction: public CategoricalCrossEntropy {
    public:
    double calculate(
        std::vector<std::vector<double>> output,
        std::vector<int> y
    );
};