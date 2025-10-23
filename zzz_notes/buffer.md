i need to do the back propagation and reduce the loss for a single neuron;


====


1.do the backpropagation on code (ie without matrix);
2.use matrices instead of vectors;
3.be able to use a single neuron in the code (ie right now
my basic building block is layer);

continue from
https://youtu.be/cuveqaYX1bw?si=bQDCwGwZsz8HNo3m&t=1233

=======

 L = y^2     y = a1 + a2 + a3
 a1 = Relu(z1)      a2 = Relu(z2)       a3 = Relu(z3)
 z1 = w11x1 + w12x2 + w13x3 + w14x4 + b1
 δL/δw11 = δL/δy * δy/δa1 * δa1/δz1                     * δz1/δw11
         = 2y    *  1     *  (1 when z1 > 0 else 0)     * x1

===
take dL/dz is a row matrix,
so for a batch size 3 , dL/dz = 3*4


==
i think its right time to create a Mat;

add backward in relu,
it needs dl_da as input; find dl_dz

======
======
======
pending:
improve row(int i)
adding begin and end might make it ready for stl
understand ratio of learning rate and dl_dw, and may be keeping that ration in optimizer class side might be a better choice
make optimizer defaults as 0;, ie to disable features

objective meaning of loss

below section feels like mess in the latest glossary
mynn::Mat avoid allocatiions say implace adding and getweightref etc

===

 think without decay i got better accuracy:
but with decay there is no fluctuation


<=====================>

design decisions made.

1d matrixes are usually row matrix
made layers mostly stateless
so had to make optimizer stateful

<========>

1.think if keeping a default constructor could cause issues to any logic
any why all member variables has to be initialized in constructor?

2. lets follow the concept followed in video

input in single row;
weight of a neuron in single row;

===
how does relu backward gets the input?,
same problem is there in all backwards;

== 
loss doesnt have to have forward, calculate is better name ig


<=====>

denseLayer
create a random value matrix

==
find accuracy from anotteher dataset

===

lec30:
till now in the code the loss, was cross entropy loss, why suddenly change it to (predicted-actual) ^ 2?
avoid 2 denselayerconstructors
implement dropout layer code

try k fold validation
test if dropout has improved testing accuracy with 1000 dataset;