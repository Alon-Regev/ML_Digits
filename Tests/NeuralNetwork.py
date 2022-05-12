import numpy
import random

LEARNING_RATE = 0.02

def main():
    # layers: [2 inputs, 4 hidden, 2 output]
    layers_sizes = [2, 4, 1]
    weights_l1 = numpy.random.rand(layers_sizes[0], layers_sizes[1])
    weights_l2 = numpy.random.rand(layers_sizes[1], layers_sizes[2])
    biases_l1 = numpy.random.rand(layers_sizes[1])
    biases_l2 = numpy.random.rand(layers_sizes[2])

    def get_data():
        inputs = numpy.array([random.random() * 1/3 + 1/12 for _ in range(layers_sizes[0])])
        outputs = numpy.array([inputs[0] + inputs[1]])
        return (inputs, outputs)

    def get_output(inp):
        outputs_l1 = sigmoid(numpy.dot(inp, weights_l1) + biases_l1)
        outputs_l2 = sigmoid(numpy.dot(outputs_l1, weights_l2) + biases_l2)
        return outputs_l2

    for _ in range(500000):
        # front propagation
        inputs, real_outputs = get_data()

        f_l1 = numpy.dot(inputs, weights_l1) + biases_l1
        outputs_l1 = sigmoid(f_l1)
        f_l2 = numpy.dot(outputs_l1, weights_l2) + biases_l2
        outputs_l2 = sigmoid(f_l2)

        # back propagation
        # out = sigmoid(f) = sigmoid(sum(weight * input) + bias)
        # E = sum((output - real)**2)
        # layer 2: dE/dp = dE/dout * dout/df * df/dp
        # layer 1: dE/dp = dE/dout_l2 * dout_l2/df_l2 * df_l2/dout_l1 * dout_l1/df_l1 * df_l1/dp

        # layer 2
        dEdf_l2 = 2 * sum(outputs_l2 - real_outputs) * sigmoid_derivative(f_l2)
        dEdb_l2 = dEdf_l2
        dEdw_l2 = numpy.outer(outputs_l1, dEdf_l2)
        dEdo_l1 = numpy.dot(dEdf_l2, weights_l2.T)  # transpose (multiplying backwards)
        # layer 1
        dEdf_l1 = dEdo_l1 * sigmoid_derivative(f_l1)
        dEdb_l1 = dEdf_l1
        dEdw_l1 = numpy.outer(inputs, dEdf_l2)

        # update weights and biases
        weights_l1 -= LEARNING_RATE * dEdw_l1
        weights_l2 -= LEARNING_RATE * dEdw_l2
        biases_l1 -= LEARNING_RATE * dEdb_l1
        biases_l2 -= LEARNING_RATE * dEdb_l2

    print(weights_l1)
    print(biases_l1)
    print(weights_l2)
    print(biases_l2)

    # calculate error
    error = 0
    for _ in range(1000):
        d = get_data()
        error += sum((get_output(d[0]) - d[1]) ** 2)
    print("Error:", error / 1000)

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