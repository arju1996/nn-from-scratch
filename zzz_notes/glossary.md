
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

===
Underfitting = The model is too simple and didn't learn enough.

Overfitting = The model memorized the specific examples.
or is what happens when your model learns the training data too well, confusing specific details for general rules

Generalization is how well your machine learning model can apply what it learned to new, unseen data.
or optimal

===
actually we should choose hyper parameter based on a validation data, and if we have limited data.
take validation from training data itself,
ie split data into k(say 5) parts
take first part as validation data, find validation error
take second part as val data, find val error
..
take 5th part as val data, find val error
find mean validation error
ie mean validatn error = 1/5 * (E1 + E2 + .. + E5)

find mean VE for multiple set of hyper params,
choose the set which gave lowest mean
and train

k fold cross validation:

===
data leakage, testing somehow leaks into training.
as it already provides info about result;

===
l1 l2 regularization: / or penalty on weights
usually bigger weights and bias means overfitting
and smaller params means good generalization

l1 loss += lambda * sum of all, abs (weights) 
l2 loss += lambda * sum of all, weights ^ 2 
l2 is usually prefered, cause of low penalyy for smaller weights

==
dl1w_dwm = lambda * {1 when dwm is +ve/0 or -1 when dwm is -ve}
dl2w_dwm = 2 lambda wm
l1 l2 regularization works better when training data is alot.

===