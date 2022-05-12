import numpy
import random
import matplotlib.pyplot as plt

LEARNING_RATE = 0.004

def main():
    # layers: [2 inputs, 4 hidden, 1 output]
    layers_sizes = [2, 2]
    weights = numpy.zeros(shape=(layers_sizes[0], layers_sizes[1]))
    biases = numpy.zeros(shape=(layers_sizes[1]))

    def get_data():
        inputs = [random.random() * 1/3 + 1/12 for _ in range(layers_sizes[0])]
        outputs = [inputs[0] + inputs[1], 4 * inputs[0] * inputs[1]]
        return (inputs, outputs)

    def get_output(inp):
        return sigmoid(numpy.dot(inp, weights) + biases)

    for iteration in range(500000):
        # calculate derivatives
        # f = sum(weight * i) + bias
        # output = s(f)
        # error = sum((output - y) ^ 2
        # dE/dp = dE/dout * dout/df * df/dp
        #       = 2 * (output - y) * s'(f) * df/dp
        # dE/dw = 2 * (output - y) * s'(f) * inp
        # dE/db = 2 * (output - y) * s'(f)

        # for each output
        inp, real_out = get_data()
        f = numpy.dot(inp, weights) + biases
        dsigmoid_f = sigmoid_derivative(f)
        outputs = sigmoid(f)
        # update parameters
        db = 2 * (outputs - real_out) * dsigmoid_f
        dw = 2 * dsigmoid_f * numpy.outer(outputs - real_out, inp)
        biases -= db * LEARNING_RATE
        weights -= dw * LEARNING_RATE

    # calculate error
    error = 0
    for _ in range(1000):
        d = get_data()
        error += sum((get_output(d[0]) - d[1]) ** 2)
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