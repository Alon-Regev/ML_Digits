from NeuralNetwork import NeuralNetwork
import numpy as np

def main():
    nn = NeuralNetwork([2, 4, 1])

    # generate data
    random_pairs = np.random.rand(1000, 2) / 3 + 1/12
    data = [(pair, np.array([sum(pair)])) for pair in random_pairs]

    print("Error: ", nn.test(data))
    for _ in range(10):
        nn.fit(data, 10000)
        print("Error: ", nn.test(data))

    while True:
        n1 = float(input("Enter first number: "))
        n2 = float(input("Enter second number: "))
        print("Result: ", nn.predict(np.array([n1, n2])))

if __name__ == "__main__":
    main()