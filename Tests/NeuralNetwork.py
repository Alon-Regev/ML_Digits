import numpy
import random
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1

def main():
    # layers: [2 inputs, 4 hidden, 1 output]
    layers_sizes = [2, 1]
    weights = [0, 0]
    bias = 0

    def get_data():
        inputs = [random.random() / 3 + 1/12 for _ in range(2)]
        output = sum(inputs)
        return (inputs, output)

    def get_output(inp):
        return sigmoid(sum([weights[j] * inp[j] for j in range(2)]) + bias)

    for iteration in range(50000):
        # calculate derivatives
        # f = sum(weight * i) + bias
        # output = s(f)
        # error = sum((output - y) ^ 2
        # dE/dp = dE/dout * dout/df * df/dp
        #       = 2 * (output - y) * s'(f) * df/dp
        # dE/dw = 2 * (output - y) * s'(f) * x
        # dE/db = 2 * (output - y) * s'(f)

        # get output
        inp, out = get_data()
        f = sum([weights[j] * inp[j] for j in range(2)]) + bias
        sdf = sigmoid_derivative(f)
        output = sigmoid(f)
        # update parameters
        db = 2 * (output - out) * sdf
        bias -= db * LEARNING_RATE
        for j in range(len(weights)):
            weights[j] -= db * inp[j] * LEARNING_RATE

    # calculate error
    error = 0
    for _ in range(1000):
        d = get_data()
        error += (get_output(d[0]) - d[1]) ** 2
    print("Error:", error)

    # play with result
    while True:
        a = float(input("Enter in1: "))
        b = float(input("Enter in2: "))
        print("result:", get_output([a, b]))
    

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * sigmoid(1 - x)

if __name__ == "__main__":
    main()