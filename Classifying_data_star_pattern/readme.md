# Star Problem
## Goals:
1. Designing MLP without using any framework.
2. Coding backpropagation algorithm.
3. Understanding use of different loss functions.

### Description of my code and network:

- I am using a single hidden layer network with 4 neurons in the hidden
layer and 2 neurons in the last layer as I am using one hot encoding.
- I found a good initialization of weights by randomly generating weights
and experimenting and I have used those weights for initialising my
network for all the experiments (unless mentioned otherwise) here on.
- I have used a sigmoid activation function in my neurons.
- My cost function is simple: , where y is predicted J = y − t output and t
is true value.
- I have written my code in a Jupyter notebook and I have 2 training
blocks followed by a testing block/cell:
  1. For the dataset provided in the problem statement
  2. For the dataset containing some randomly generated points in the
square size 2 centered at origin.

- My best results are below:
I get best results with a certain set of weights and learning rate
between 0.4 and 1.


### Number of neurons in the hidden layer:

I have used 4 neurons in my hidden layer, there are a couple of reasons
behind this choice:

**1. Distribution of the data:**
The data given in the problem is class 1 near the x/y axes and 0 everywhere
else so, if we try to distinguish them by hand it can be done accurately with 4
lines, 2 on each side of x axis and 2 on each side of y axis.
Thus 4 neurons, each for one line.

**2. Experimentation:**
I experimented with different numbers of neurons in the hidden layer and I
chose 4 based on the results. While I agree that there are other factors like
weight initialization and learning rate, I found the best hyperparameters
through experiment (learning rate of 0.4) and used them with different
numbers of neurons in the hidden layer.
