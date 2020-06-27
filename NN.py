import math
import numpy as np
import random
Training_data = [
    {'inputs': [0, 1],'target': [1]},
    {'inputs': [1, 0],'target': [1]},
    {'inputs': [0, 0],'target': [0]},
    {'inputs': [1, 1],'target': [0]}
]

class NeuralNetwork(object):
    
    def __init__(self, input_layer, hidden_layer, output_layer):
        
        self.input_nodes = input_layer
        self.hidden_nodes = hidden_layer
        self.output_nodes = output_layer

        # activation function
        self.sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
        
        # derivative of activation function
        self.dsigmoid = np.vectorize(lambda x: x * (1 - x))

        self.learning_rate = 0.1

        # the weights of input to hidden layer connections
        self.weights_ih = np.random.uniform(-1, 1, (self.hidden_nodes, self.input_nodes))
        # the weights of hidden to output layer connections
        self.weights_ho = np.random.uniform(-1, 1, (self.output_nodes, self.hidden_nodes))

        # biases of the hidden layer
        self.bias_h = np.random.rand(self.hidden_nodes, 1)
        # biases of the output layer
        self.bias_o = np.random.rand(self.output_nodes, 1)


    def feedforward(self, input_array):

        # the input is an array, we need to change it into a vector
        input_vector = np.asmatrix(input_array)
        input_vector = np.reshape(input_vector, (self.input_nodes, 1))

        # Generating the hidden layer results
        hidden = self.weights_ih.dot(input_vector)
        hidden = hidden + self.bias_h

        # run the weighted sums through the activation function
        hidden = self.sigmoid(hidden)

        # Generating the output layer results
        output = self.weights_ho.dot(hidden)
        output = output + self.bias_o

        # run the weighted sums through the activation function
        output = self.sigmoid(output)

        return output

    def train(self, input_array, answers_array):

        # the input is an array, we need to change it into a vector
        input_vector = np.asmatrix(input_array)
        input_vector = np.reshape(input_vector, (self.input_nodes, 1))

        # Generating the hidden layer results
        hidden = self.weights_ih.dot(input_vector)
        hidden = hidden + self.bias_h
        hidden = self.sigmoid(hidden)
        
        # Generating the output layer results
        output = self.weights_ho.dot(hidden)
        output = output + self.bias_o
        output = self.sigmoid(output)

        # the answers is an array, we need to change it into a vector
        answers_vector = np.asmatrix(answers_array)
        answers_vector = np.reshape(answers_vector, (self.output_nodes, 1))

        # calculate the GENERAL error
        output_errors = answers_vector - output

        # calculate output gradient
        # Formula: LR * E * d(final_output) . hidden_layer_values_transposed
        derivative_output = self.dsigmoid(output)
        
        # Element-wise multiplcation
        weights_gradient = np.multiply(derivative_output, output_errors)
        weights_gradient = weights_gradient * self.learning_rate

        hidden_t = hidden.transpose()
        weights_ho_deltas = weights_gradient.dot(hidden_t)
        
        # adjust the weights
        self.weights_ho = self.weights_ho + weights_ho_deltas
        self.bias_o = self.bias_o + weights_gradient

        # calculate the hidden layer error
        weights_ho_t = self.weights_ho.transpose()
        hidden_errors = weights_ho_t.dot(output_errors)

        # Calculate hidden gradient
        # Formula: LR * E * d(final_output) . hidden_layer_values_transposed
        derivative_output = self.dsigmoid(hidden)

        # Element-wise multiplcation
        hidden_gradient = np.multiply(derivative_output, hidden_errors)
        hidden_gradient = hidden_gradient * self.learning_rate
        
        inputs_t = input_vector.transpose()
        weights_ih_deltas = hidden_gradient.dot(inputs_t)

        # adjust input -> hidden weights
        self.weights_ih = self.weights_ih + weights_ih_deltas
        self.bias_h = self.bias_h + hidden_gradient
  

if __name__ == "__main__":
    nn = NeuralNetwork(2, 4, 1)

    # train the neural network 10,000 times
    for i in range(10000):
        el = random.choice(Training_data)
        nn.train(el['inputs'], el['target'])
    
    print('0 XOR 0 = {}'.format(nn.feedforward([0, 0])))
    print('0 XOR 1 = {}'.format(nn.feedforward([0, 1])))
    print('1 XOR 0 = {}'.format(nn.feedforward([1, 0])))
    print('1 XOR 1 = {}'.format(nn.feedforward([1, 1])))
