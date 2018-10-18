import numpy as np


class Neural_Network():

    def __init__(self, num_features, num_outputs):
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.weights = {}

    def set_learning_rate(self,learning_rate):
        self.learning_rate = learning_rate

    def initialization_weights_biases(self):
        self.weights['Hidden_layer_weights'] = np.random.uniform(-1, 1, (self.num_features, self.num_features))
        self.weights['Biases_hidden_layer'] = np.random.uniform(-1, 1, (self.num_features, 1))

        self.weights['output_layer_weights'] = np.random.uniform(-1, 1, (self.num_features, self.num_outputs))
        self.weights['Biases_output_layer'] = np.random.uniform(-1, 1, (self.num_outputs, 1))

    def sigmoid(self, linear_output):
        return 1.0 / (1.0 + (np.exp(-linear_output)))

    def sigmoid_derivative(self, output):
        return output * (1 - output)

    def cross_entropy(self, y, output):
        return -y * np.log(output) - (1 - y) * np.log(1 - output)

    def feed_forward(self, X):
        self.hidden_layer_linear = np.add(np.dot(self.weights['Hidden_layer_weights'], X),
                                          self.weights['Biases_hidden_layer'])
        self.hidden_activation = self.sigmoid(self.hidden_layer_linear)
        self.output_layer_linear = np.add(
            np.dot(np.transpose(self.weights['output_layer_weights']), self.hidden_activation),
            self.weights['Biases_output_layer'])
        output_activation = self.sigmoid(self.output_layer_linear)
        return output_activation

    def backprop(self, X, target, output):
        delta = (output - target)
        hidden_layer_delta = np.dot(self.weights['output_layer_weights'], delta) * self.sigmoid_derivative(
            self.hidden_activation)
        input_error = np.dot(hidden_layer_delta, np.transpose(X))
        self.weights['output_layer_weights'] -= self.learning_rate * np.dot(self.hidden_activation, delta)
        self.weights['Hidden_layer_weights'] -= self.learning_rate * input_error
