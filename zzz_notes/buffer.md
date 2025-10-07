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