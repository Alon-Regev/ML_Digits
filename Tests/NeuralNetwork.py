import numpy
import random
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01

def main():
    # layers: [2 inputs, 4 hidden, 1 output]
    layers_sizes = [1, 1]
    weight = 0
    bias = 0

    data = {(x / 100): (x / 200) + random.random() / 50 for x in range(50)}

    def get_output(input):
        return weight * input + bias

    for iteration in range(10000):
        # calculate derivatives
        # output = weight * x + bias
        # error = (output - y) ^ 2
        # dE/dp = 2 * (output - y) * df/dp
        # dE/dw = 2 * (output - y) * x
        # dE/db = 2 * (output - y)

        # get output
        x, y = random.choice(list(data.items()))
        output = get_output(x)
        dw = 2 * (output - y) * x
        db = 2 * (output - y)
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

if __name__ == "__main__":
    main()