# DEEP LEARNING TWO LAYERS NEURAL NETWORK

## INTRODUCTION
The purpose of this project is to understand how both feedforward and back propagation algorithms work inside a neural network.
This simple network was implemented in a naive, not optimized way, for educational purposes.

## PROJECT DESCRIPTION
In this project we will simulate the behavior of an XOR operator using a two layers neural network.

![XOR!](http://hyperphysics.phy-astr.gsu.edu/hbase/Electronic/ietron/xor.gif)
![Two layers NN](https://i.ibb.co/GpMhztV/Untitled-Diagram-1.png)

Since XOR is not a [`linearly separable`](https://en.wikipedia.org/wiki/Linear_separability) problem, meaning we cannot draw a single line that will classify our data, we will need more than one neuron to find a solution.

This is a visual representation of the solution space.

![learning_perceptron](https://i.ibb.co/TYg0s18/ezgif-4-5292c30bbeea.gif)

## PROJECT STRUCTURE
`feedforward` Will take input data and return the result as weighted sums.

`train` Will train the network and change weights each iteration, also known as [`Stochastic gradient descent`](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

Note:
Since everything is randomized in this project, the results will differ with each run.
