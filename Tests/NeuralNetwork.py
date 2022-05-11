import numpy
import random
import matplotlib.pyplot as plt

LEARNING_RATE = 0.5

def main():
    # layers: [2 inputs, 4 hidden, 1 output]
    layers_sizes = [1, 1]
    weight = 0
    bias = 0

    data = {(x / 10 - 8): sigmoid(x / 20 - 5 + random.random()) for x in range(200)}

    def get_output(input):
        return sigmoid(weight * input + bias)

    for iteration in range(100000):
        # calculate derivatives
        # f = weight * x + bias
        # output = s(f)
        # error = (output - y) ^ 2
        # dE/dp = dE/dout * dout/df * df/dp
        #       = 2 * (output - y) * s'(f) * df/dp
        # dE/dw = 2 * (output - y) * s'(f) * x
        # dE/db = 2 * (output - y) * s'(f)

        # get output
        x, y = random.choice(list(data.items()))
        f = weight * x + bias
        sdf = sigmoid_derivative(f)
        output = sigmoid(f)
        db = 2 * (output - y) * sdf
        dw = db * x
        # update parameters
        weight -= dw * LEARNING_RATE
        bias -= db * LEARNING_RATE

    # calculate error
    error = 0
    for x, y in data.items():
        error += (get_output(x) - y) ** 2
    print(error)

    # plot
    x_values = list(data.keys())
    y_values = list(data.values())
    plt.scatter(x_values, y_values)
    plt.plot(x_values, [get_output(x) for x in x_values], color="orange")
    plt.show()

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * sigmoid(1 - x)

if __name__ == "__main__":
    main()