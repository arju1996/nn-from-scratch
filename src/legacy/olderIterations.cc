#include <iostream>
#include <array>
#include <vector>

#include <cassert>

#define INPUTSIZE 4

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
    Node(int inputsize,  float bias, std::vector<float> weight): weight(weight), bias(bias) {
        // i could add some validations here
        // like input size and weight size must be same.
        // and add the same validations in the run part as well;
    };
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
    int inputsize, noofnodes;
    std::vector<NodeFinal> nodes;
    std::vector<Node> nodez;
    public:
    Layer(std::vector<NodeFinal> nodes) : nodes(nodes) {}
    Layer(int inputsize, int noofnodes, std::vector<Node> nodes) : 
        inputsize(inputsize), 
        noofnodes(noofnodes), 
        nodez(nodes) {
        for(int i = 0; i < inputsize; ++i) {
            //
        }
    }


    // np.dot(input, w) + bias
    // np.dot( w, input) + bias
    // both are valid
    std::vector<float> run(std::vector<float> input) {
        assert(input.size() == inputsize);
        std::vector<float> result;
        for(auto node: nodez) {
            float resultpernode = node.run(input);
            result.push_back(resultpernode);
        }
        return result;
    }

    // so this naming doesnt make sense;
    // as i can only do dot product on matrices;
    // delete this
    std::vector<float> runDotProduct(std::vector<float> input) {
        assert(input.size() == inputsize);
        std::vector<float> result;
        for(auto node: nodez) {
            float resultpernode = node.run(input);
            result.push_back(resultpernode);
        }
        return result;
    }


    // np.dot(inputs, w^T) + bias
    std::vector<std::vector<float>> runInputBatch(std::vector<std::vector<float>> inputBatch, int batchsize) {
        assert(inputBatch.size() == batchsize);
        // assert(input.size() == inputsize);
        std::vector<std::vector<float>> result;
        for(int i=0; i<batchsize; ++i) {
            std::vector<float> resultperbatch = runDotProduct(inputBatch.at(i));
            result.push_back(resultperbatch);
        }
        return result;
    }
};

class NN {
    std::vector<Layer> layers;
    public:
    NN(std::vector<Layer> layers) : layers(layers) {

    }
    std::vector<float> run(std::vector<float> input) {
        std::vector<float> result = input;
        for(auto layer: layers) {
            result = layer.run(result);
        }
        return result;
    }

    std::vector<std::vector<float>> runInputBatch(std::vector<std::vector<float>> inputBatch, int batchsize) {
        std::vector<std::vector<float>> result;
        for(int i=0; i<batchsize; ++i) {
            std::vector<float> resultperbatch = run(inputBatch.at(i));
            result.push_back(resultperbatch);
        }
        return result;
    }

};

// i gotta remove the 4, no of inputs from node and Layers,
// ways at which we create the layers can be improved, like it can be a matrix;

int main() {
     int inputsize = 4;
    int nodecount = 3;
    Layer l1 {
        inputsize,
        nodecount,
        {
            // NodeFinal{{0.2, 0.3, 0.4, 0.4}, 0.2},
            // NodeFinal{{0.1, 0.2, 0.3, 0.1}, 0.3},
            // NodeFinal{{0.4, 0.1, 0.1, 0.2}, 0.4}

            // Node{inputsize, 0.1, {0.1, 0.1, 0.1}},
            // Node{inputsize, 0.2, {0.2, 0.2, 0.1}},

            Node{inputsize, 2, {0.2, 0.8, -0.5, 1}},
            Node{inputsize, 3, {0.5, -0.91, 0.26, -0.5}},
            Node{inputsize, 0.5, {-0.26, -0.27, 0.17, 0.87}},
        }
    };
    std::cout<<"hello world";
    std::vector<float> result = l1.run({1, 2, 3, 2.5});
    /*
    std::cout<<"\n\n\nresult";
    for (auto a: result) {
        std::cout << a<< " ";
    }
    */

    std::vector<std::vector<float>> result2= l1.runInputBatch({{1, 2, 3, 2.5}, {2, 5, -1, 2}, {-1.5, 2.7, 3.3, -0.8}}, 3);
    /*
    std::cout<<"\n\n\nresult2";
    for (auto result: result2) {
        for (auto a: result) {
            std::cout << a<< " ";
        }
        std::cout<<"\n";
    }
    */

    inputsize = 3;
    inputsize = 3;
    Layer l2 {
        inputsize,
        nodecount,
        {
            Node{inputsize, -1, {0.1, -0.14, 0.5}},
            Node{inputsize, 2, {-0.5, 0.12, -0.33}},
            Node{inputsize, -0.5, {-0.44, 0.73, -0.13}},
        }
    };

    NN nn {{l1, l2}};
    std::vector<float> result3 = nn.run({1, 2, 3, 2.5});
    std::cout<<"\n\n\nresult3 ";
    for (auto a: result3) {
        std::cout << a<< " ";
    }

    std::vector<std::vector<float>> result4 = nn.runInputBatch({{1, 2, 3, 2.5}, {2, 5, -1, 2}, {-1.5, 2.7, 3.3, -0.8}}, 3);
    std::cout<<"\n\n\nresult4";
    for (auto result: result4) {
        for (auto a: result) {
            std::cout << a<< " ";
        }
        std::cout<<"\n";
    }

}