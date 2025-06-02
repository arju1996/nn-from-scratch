#include <iostream>
#include <array>
#include <vector>

#define INPUTSIZE 3

class NodeBasic {
    std::array<int, 3> input;
    std::array<int, 3> weight;
    int bias;
    float output;
    NodeBasic(std::array<int, 3> input, std::array<int, 3> weight, int bias) : input(input), weight(weight), bias(bias){

    }

    float run() {
        output = 0.0f;
        for(int i = 0 ; i < input.size(); ++i) {
            output += input[i]*weight[i];
        }
        output += bias;
    }
};

// code in such a way that 4 inputs and 3 nodes in a later;
class Node {
    std::vector<float> weight;
    float bias;
    public:
    Node(int inputsize,  float bias, std::vector<float> weight): weight(weight), bias(bias) {};
    float run (std::vector<float> input) {
        float result = 0.0f;
        for (int i = 0; i < weight.size(); ++i) {
            result += weight.at(i) * input.at(i);
        }
        result += bias;
        return result;
    }
};

class NodeFinal {
    std::array<float, INPUTSIZE> weight;
    float bias;
    public:
    NodeFinal(std::array<float, INPUTSIZE> weight, float bias): weight(weight), bias(bias) {};
    float run (std::array<float, INPUTSIZE> input) {
        float result = 0.0f;
        for (int i = 0; i < weight.size(); ++i) {
            result += weight.at(i) * input.at(i);
        }
        result += bias;
        return result;
    }
};

// each layer can have any many nodes as it wants;
class Layer {
    std::vector<Node> nodes;
    public:
    Layer(std::vector<Node> nodes) : nodes(nodes) {}
};

// i gotta remove the 4, no of inputs from node and Layers,
// ways at which we create the layers can be improved, like it can be a matrix;

int main() {
    Layer l1 {
        {
            Node{4, 0.2, {0.2, 0.3, 0.4, 0.4}},
            Node{4, 0.3, {0.1, 0.2, 0.3, 0.1}},
            Node{4, 0.4, {0.4, 0.1, 0.1, 0.2}}
            // Node{3, 0.1, {0.1, 0.1, 0.1}},
            // Node{3, 0.2, {0.2, 0.2, 0.1}},
        }
    };
    std::cout<<"hello world";
    return 0;
}