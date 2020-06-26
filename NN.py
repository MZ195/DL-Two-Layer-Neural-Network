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

        self.learning_rate = 0.1

        self.weights_ih = np.random.uniform(-1, 1, (self.hidden_nodes, self.input_nodes))
        self.weights_ho = np.random.uniform(-1, 1, (self.output_nodes, self.hidden_nodes))


        self.bias_h = np.random.rand(self.hidden_nodes, 1)
        self.bias_o = np.random.rand(self.output_nodes, 1)


    def feedforward(self, input_array):

        # Generating the hidden layer results
        local_input = np.asmatrix(input_array)
        local_input = np.reshape(local_input, (self.input_nodes, 1))

        hidden = self.weights_ih.dot(local_input)
        hidden = hidden + self.bias_h

        # Activation function
        sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
        hidden = sigmoid(hidden)
        

        # Generating the output layer results
        output = self.weights_ho.dot(hidden)
        output = output + self.bias_o

        # Activation function
        output = sigmoid(output)

        # send back as array
        return output

    def train(self, input_array, answers_array):
        # Activation function
        sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
        # derivative of sigmoid    
        dsigmoid = np.vectorize(lambda x: x * (1 - x))


        local_input = np.asmatrix(input_array)
        local_input = np.reshape(local_input, (self.input_nodes, 1))

        # Generating the hidden layer results
        hidden = self.weights_ih.dot(local_input)
        hidden = hidden + self.bias_h
        hidden = sigmoid(hidden)
        
        # Generating the output layer results
        output = self.weights_ho.dot(hidden)
        output = output + self.bias_o
        output = sigmoid(output)

        answers_array = np.asmatrix(answers_array)
        answers_array = np.reshape(answers_array, (self.output_nodes, 1))

        # calculate the GENERAL error
        output_errors = answers_array - output

        #calculate output gradient
        # Formula: LR * E * d(final_output) . hidden_layer_values_transposed
        derivative_output = dsigmoid(output)
        # Element-wise multiplcation
        weights_gradient = np.multiply(derivative_output, output_errors)
        hidden_t = hidden.transpose()
        weights_gradient = weights_gradient * self.learning_rate
        weights_ho_deltas = weights_gradient.dot(hidden_t)

        # adjust the weights
        self.weights_ho = self.weights_ho + weights_ho_deltas
        self.bias_o = self.bias_o + weights_gradient

######################################################################################################
        #WOKING ON HIDDEN LAYERS

        # Calculate hidden gradient
        weights_ho_t = self.weights_ho.transpose()
        hidden_errors = weights_ho_t.dot(output_errors)
        derivative_output = dsigmoid(hidden)

        # Element-wise multiplcation
        hidden_gradient = np.multiply(derivative_output, hidden_errors)
        inputs_t = local_input.transpose()
        hidden_gradient = hidden_gradient * self.learning_rate
        weights_ih_deltas = hidden_gradient.dot(inputs_t)

        # adjust input -> hidden weights
        print(self.weights_ih)
        self.weights_ih = self.weights_ih + weights_ih_deltas
        self.bias_h = self.bias_h + hidden_gradient
        print(self.weights_ih)

        print('\n')

if __name__ == "__main__":
    nn = NeuralNetwork(2, 2, 1)


    for i in range(10):
        el = random.choice(Training_data)
        nn.train(el['inputs'], el['target'])
    
    # print(nn.feedforward([1, 1]))
