# Creating "A" by using perceptrons.
I approached drawing A by drawing the lines, writing their equations and then
deciding the weights of the nodes, I have men described my solving process below:
### Specifications of network:
- Activation function: Signum
- Number of hidden layers: 2
- Number of nodes in first hidden layer: 6
- Number of nodes in second hidden layer: 3
- Total classes: 2 (1 for black and -1 for white)
- The center of the “A” I got is at (0, 0) but you can choose any size of the grid.
### Explanation for number of nodes in first hidden layer and weights:
I drew 6 lines which form the A, the equations of line in slope-intercept form
are below:
Please note that there are 2 lines which sort of bound the white region.
1. Lines on the right:
  - Line 1 (L 1 ): y = − 3.0 x + 3
  - Line 2 (L 2 ): y = − 3.0 x + 2.5
2. Lines on the left:
  - Line 3 (L 3 ): y = 3.0 x + 3
  - Line 4 (L 4 ): y = 3.0 x + 2.5
3. Center lines:
  - Line 5 (L 5 ): y = − 1.0
  - Line 6 (L 6 ): y = − 1.25
  
So, I have used 6 nodes in the first hidden layer. The weights which I got from
the equations of these lines are as follows:
Weights are in the format: “Node name: [w1, w2, b] ”, where w1 is the weight
for the input x and w2 is the weight for the input y and b is the bias.

- Layer 1:
  - N11 node: [− 3.0, 1.0, − 3.0]
  - N12 node: [− 3.0, 1.0, − 2.5]
  - N13 node: [3.0, 1.0, − 3.0]
  - N14 node: [3.0, 1.0, − 2.5]
  - N15 node: [0.0, 1.0, 1.0]
  - N16 node: [0.0, 1.0, 1.25]

Where N11 to N16 are the nodes for the lines I have defined above.

### Number of nodes in second hidden layer and weights:

Now for the second hidden layer as we need to divide the region so as to get
an “A” like structure.
The logic I used for deciding the weights for connections from first hidden
layer to second hidden layer is that the output from a node is either 1 or -1
(because of signum activation function) and as the nodes in the first layer
represent lines so, it’s like a line is dividing a region into +1 and -1. 
Now to combine all of them I have used the following weights which I got by
doing some trials.

- N21 node: [− 1.0, 1.0, − 1.0, 0.0, − 1.0, 1.0, − 2.0]
- N22 node: [− 1.0, 0.0, − 1.0, 1.0, − 1.0, 1.0, − 2.0]
- N23 node: [− 1.0, 1.0, − 1.0, 1.0, 1.0, − 1.0, − 1.0]

Where, N21 , N22 and N23 are the nodes of the second hidden layer.

### Final (Output) layer:

For final layer weights are simply all 1 as we already have the regions
for ”A” from the second hidden layer so, we don’t need to multiply with any
weights in this layer we just want to combine them here.
- Output node: [1, 1, 1, 0]

### Achieving this with a single hidden layer when Black = 0 and white = 1 is also possible. 

I achieved this with 12 nodes in the hidden layer. Below is the
figure I got with 1 layer (code is in the last cell of the notebook):

The first 6 nodes are the same as in the previous 2 hidden layer’s first layer
and the next 6 nodes are to compensate for the effects of the second hidden
layer in the previous part. Below is my single layer:

 - [[[− 3.0, 1.0, − 3.0], [− 3.0, 1.0, − 2.5], [3.0, 1.0, − 3.0], [3.0, 1.0, − 2.5],[0.0, 1.0, 1.0], [0.0, 1.0, 1.25], [1.0, 0.0, 1.39], [1.0, 0.0, x_value], [1.0, 0.0, − 1.39],
[1.0, 0.0, x_neg_value], [0.0, 1.0, y_neg_value], [0.0, 1.0, − 2.75]],[[− 2.0, 2.0, − 2.0, 2.0, − 1.0, 1.0, 1.0, − 1.0, − 1.0, 1.0, 2.0,− 2.0, − 1.0]]]

Where, x_neg_value is the negative x limit of the plot and y_neg_value is the
negative y limit of the plot and x_value is the positive x limit of the plot.

I changed the activation function for this part by changing the
second class from -1 to 0.
