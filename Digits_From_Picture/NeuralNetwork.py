import numpy as np
import random

LEARNING_RATE = 0.01

class NeuralNetwork():
    def __init__(self, layers_sizes: list, learning_rate: float = LEARNING_RATE):
        self.layers_sizes = layers_sizes
        self.layer_count = len(layers_sizes) - 1
        self.learning_rate = learning_rate
        # initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.layer_count):
            self.weights.append(np.random.rand(layers_sizes[i], layers_sizes[i + 1]) / 2 - 0.25)
            self.biases.append(np.random.rand(layers_sizes[i + 1]) / 2 - 0.25)
        self.activation_functions = [sigmoid, softmax]
        self.activation_derivatives = [sigmoid_derivative, softmax_derivative]

    def fit(self, training_data: list, iterations: int = 1000) -> None:
        """
        method runs iterations to fit neural network to dataset.
        can be called multiple times to continue training.
        input:
            training_data: list of tuples of np arrays (inputs, desired output)
            iterations: number of iterations to train for.
        return: None
        """
        for _ in range(iterations):
            # front propagation
            data = random.choice(training_data)
            f_l, o_l = [], [data[0]]
            # get output of each layer
            for i in range(self.layer_count):
                f_l.append(o_l[i - 1].dot(self.weights[i]) + self.biases[i])
                o_l.insert(i, self.activation_functions[i](f_l[-1]))
            
            # back propagation
            # derivative of last layer (based on error function)
            dEdo = 2 * (o_l[self.layer_count - 1] - data[1])
            # go over layers backwards
            for i in range(self.layer_count)[::-1]:
                # calculate derivatives
                dEdf = dEdo * self.activation_derivatives[i](f_l[i])
                dEdb = dEdf
                dEdw = np.outer(o_l[i - 1], dEdf)
                dEdo = np.dot(dEdf, self.weights[i].T)
                # update parameters
                self.biases[i] -= LEARNING_RATE * dEdb
                self.weights[i] -= LEARNING_RATE * dEdw

    def test(self, test_data: list) -> float:
        """
        method tests the error of the neural network.
        input:
            test_data: list data which doesn't appear in training (inputs, desired output)
        return: the error of the neural network
        """
        total_error = 0
        # sum error of each test entry
        for test in test_data:
            out = self.predict(test[0])
            total_error += (out - test[1]) ** 2
        # return average error
        return sum(total_error) / (len(test_data) * len(total_error))

    def predict(self, inputs: list) -> list:
        """
        method enters inputs into the current neural network and returns it's outputs.
        input:
            inputs: list of inputs (floats) to enter into the neural network
        return: outputs of neural network (list of floats)
        """
        # front propagation
        out = inputs
        for i in range(self.layer_count):
            out = self.activation_functions[i](out.dot(self.weights[i]) + self.biases[i])
        return out

    def save(self, path: str) -> None:
        """
        method saves the current neural network to a file.
        input:
            path: path to save the file to
        return: None
        """
        np.savez(path, weights=self.weights, biases=self.biases)
        
    def load(self, path: str) -> None:
        """
        method loads a neural network from a file.
        input:
            path: path to load the file from
        return: None
        """
        data = np.load(path, allow_pickle=True)
        self.weights = data['weights']
        self.biases = data['biases']

# activation function definition
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x) * sigmoid(1 - x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))
