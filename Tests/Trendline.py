import pandas
import random
import matplotlib.pyplot as plt

LEARNING_RATE = 0.0001
ITERATIONS = 5000
PLOT_POINTS = 50

PLOT_EVERY_N_ITERATIONS = ITERATIONS // (PLOT_POINTS + 1)
SKIP_ITERATIONS = 2 * PLOT_EVERY_N_ITERATIONS

def main():
    # get data from csv
    csv = pandas.read_csv("./data.csv")
    data = csv['data1'].to_dict()

    # fine trendline
    params, errors = trendline(data)

    plot_result(data, params, errors)

def trendline(data: dict) -> tuple:
    """
    uses gradient descent to find a trendline for data
    input: data - dict of points {x: y}
    return: linear function parameters over iterations list (m, b)
            error over iterations list (E)
    """
    # f(x) = mx + b
    # E(x) = sum((f(x) - y)^2)
    # dE/dp = dE/df * df/dp
    # dE/dm = 2 * sum((f(x) - y) * x)
    # dE/db = 2 * sum(f(x) - y)
    
    parameters = []
    errors = []
    # define function
    m, b = [random.random() * 2 - 1 for i in range(2)]
    f = lambda x: m * x + b
    # algorithm iterations
    for iteration in range(ITERATIONS):
        # save parameters
        parameters.append((m, b))
        errors.append(sum([(f(x) - y) ** 2 for x, y in data.items()]))
        # calculate derivatives
        db = sum([2 * (f(x) - y) for x, y in data.items()])
        dm = sum([2 * (f(x) - y) * x for x, y in data.items()])
        # move parameters down the gradient
        m -= dm * LEARNING_RATE
        b -= db * LEARNING_RATE
    
    # save final result
    parameters.append((m, b))
    errors.append(sum([(f(x) - y) ** 2 for x, y in data.items()]))
    
    return parameters, errors

def plot_result(data: dict, params: list, errors: list) -> None:
    """
    plots graph of resulting trendline
    """
    # data plot
    x1 = list(data.keys())
    y1 = list(data.values())
    plt.scatter(x1, y1, label = "data")
    
    # plot trendline
    f = lambda x: params[-1][0] * x + params[-1][1]
    x2 = [min(x1), max(x1)]
    y2 = [f(x) for x in x2]
    plt.plot(x2, y2, label = "trendline")
    
    # naming
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'y = {params[-1][0]:.3f}x + {params[-1][1]:.3f}   (E = {errors[-1]:.2f})')

    plt.show()

    # params plot
    x3, x4, y3, y4 = [], [], [], []
    for i, p in enumerate(params[SKIP_ITERATIONS::PLOT_EVERY_N_ITERATIONS]):
        x3.append(i * PLOT_EVERY_N_ITERATIONS)
        x4.append(i * PLOT_EVERY_N_ITERATIONS)
        y3.append(p[0])
        y4.append(p[1])
    # errors plot
    x5, y5 = [], []
    for i, e in enumerate(errors[SKIP_ITERATIONS::PLOT_EVERY_N_ITERATIONS]):
        x5.append(i * PLOT_EVERY_N_ITERATIONS)
        y5.append(e)
    # subplots
    f, (p1, p2) = plt.subplots(2, 1, sharex=True)
    p1.scatter(x3, y3, label="m")
    p1.scatter(x4, y4, label="b")
    p1.set_title("Params m and b over iterations")
    p2.scatter(x5, y5, label="Error")
    p2.set_title("Error over iterations")
    plt.xlabel('Iterations')

    plt.show()

if __name__ == "__main__":
    main()