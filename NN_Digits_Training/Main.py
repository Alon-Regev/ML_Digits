from mnist import MNIST
from NeuralNetwork import NeuralNetwork
import numpy as np

NeuralNetwork.LEARNING_RATE = 0.01

def main():
    # prepare training dataset and test dataset
    # get data as list of tuples (inputs, outputs)
    mndata = MNIST('.\\NN_Digits\\Dataset')
    training_data_raw = mndata.load_training()
    training_data = [(np.array(x) / 255, to_out(y)) for x, y in zip(training_data_raw[0], training_data_raw[1])]
    test_data_raw = mndata.load_testing()
    test_data = [(np.array(x) / 255, to_out(y)) for x, y in zip(test_data_raw[0], test_data_raw[1])]

    # prepare neural network
    layers_sizes = [784, 128, 10]
    nn = NeuralNetwork(layers_sizes)
    
    for i in range(10):
        nn.fit(training_data, iterations=100000)
        # test
        print(f"Iteration {i}, Error = {nn.test(test_data)}")
    nn.save(f"nn.npz")

def to_out(y):
    return (np.arange(10) == y).astype(int)

if __name__ == "__main__":
    main()