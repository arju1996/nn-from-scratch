
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


*.
decay = αt = α / {1 + d * t}

===
below section feels like mess
*.
momentum = PAST MATTERS / remember past
ie uses prev update dir to influence next update

wt = (wt-1) - (α * dl_dw) + (momentumfactor * prev weight updates)momentumfactor = 0.9 usuallly



weight_updates =  
    layerWeight * momentum_factor_ +
    dl_dw *learningRateToBeChoosen;
layerWeight = weight_updates

ie
    w = w0 - { stepsize* dl_dw + 0.9 * layerWeight }


*.adgrad

w = w - (α * dl_dw/sqrt(cache + eps))

===
sgd = 
w = w - learningrate * dl_dw

sgd with decay = 
w = w - learningrate * dl_dw
but update learning rate everyframe to
lr = lr0 / (1 + decayconst * t)

sgd with momentum = 
w = w + layerWeight;
layerWeight = -learningrate * dl_dw + momfactr * layerWeight
past matters more than present?


adagrad === idea is diff ll for diff weights 
(div by cache makes step size/change smaller)
ie change be balanced on all values;
(problem cache increases heavily easily, so stepsize very small, wights no updates)

w = w - learningrate * dl_dw / sqrt(cache + e)
cache = cache + dl_dw^2


rmsprop
uses both ideas of adagrad and momentum
millions of iterations needed to shine;

w = w - learningrate * dl_dw / sqrt(cache + e)
cache = rho * cache + (1 - rho) * dl_dw^2

because of the denominator, from the beginning itself
effectiveLR can be low

===
sgd with momentum can also be represented as
w = w - learningrate * layerWeight;
layerWeight = B1 momentum + (1 - B1) dl_dw;

===
adam
mix of 
momentum idea and cache idea

w -= learningrate * momterm/(1 - b1^t) /
     sqrt(epsilon + cacheterm/(1 - b2^t))

momterm = momterm * b1 + (1-b1) * dl_dw
cacheterm = cacheterm * b2 + (1-b2) * dl_dw^2