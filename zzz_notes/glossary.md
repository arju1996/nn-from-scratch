
*.
Dense Layer (Fully Connected Layer) == A type of layer 
where every input is connected to every output neuron.

*.
gradient is the vector of partial derivatives

*.
CategoricalCrossEntropy
a loss function for multiclass classifications
Loss = -∑(y_true * log(y_pred))


*.
Backpropagation
Loss = $(y - 0)^2$ = $Relu(wx+b) ^ 2$ = [Relu( sum( mul(w0x0) , mul(w0x0) , mul(w0x0), b) )] ^ 2
δL/δw0 = δL/δRelu * δRelu/δsum * δsum/δmul(w0x0) * δmul(w0x0)/δw0

