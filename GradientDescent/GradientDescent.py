import random
import matplotlib.pyplot as plt

PLOT_POINTS = 201

# class implements a general gradient descent algorithm
class GradientDescent:
    def __init__(self, learning_rate: float, max_iterations: int, parameter_count: int, data: dict, function: callable, derivatives: list = None):
        """
        constructor for gradient descent
        input: 
            learning_rate [int]: learning rate for gradient descent
            max_iterations [int]: number of iterations
            parameter_count [int]: number of parameters
            function [function]: function to fit to data (x, params) -> y
            derivatives [list]: list of derivatives of function (x, params) -> df/dp_i
        """
        self.function = function
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.parameter_count = parameter_count
        self.data = data
        # store history of parameters and errors
        self.parameters = []
        self.errors = []
        self.final_parameters = ()
        # set derivatives
        if derivatives is None:
            # calculate derivatives by finite differences
            self.derivatives = []
            for i in range(self.parameter_count):
                def derivative(x, params):
                    params[i] += 1e-6
                    f1 = self.function(x, params)
                    params[i] -= 1e-6
                    f2 = self.function(x, params)
                    return (f1 - f2) / 1e-6
                self.derivatives.append(derivative)
        else:
            self.derivatives = derivatives

    def error(self, parameters: list) -> float:
        """
        method to calculate error of function.
        input: 
            function [function]: function to fit to data
            parameters [list]: list of parameters
            data [dict]: points {x: y}
        return:
            error [float]: error of function
        """
        return sum([(self.function(x, parameters) - y) ** 2 for x, y in self.data.items()])
    
    def fit(self) -> tuple:
        """
        method to fit data to function.
        input: 
            data [dict]: points {x: y}
        return:
            parameters [list]: list of parameters
        """
        parameters = [random.random() * 2 - 1 for i in range(self.parameter_count)]
        self.parameters.append(tuple(parameters))
        self.errors.append(self.error(parameters))
        for iteration in range(self.max_iterations):
            # calculate derivatives
            derivatives = []
            for i in range(self.parameter_count):
                derivatives.append(
                    2 * sum([(self.function(x, parameters) - y) * 
                    self.derivatives[i](x, parameters) 
                    for x, y in self.data.items()])
                )
            # update parameters
            for i in range(self.parameter_count):
                parameters[i] -= derivatives[i] * self.learning_rate
            # save parameters and updated error
            self.parameters.append(tuple(parameters))
            self.errors.append(self.error(parameters))
        
        self.final_parameters = tuple(parameters)
        return parameters
    
    def plot(self) -> None:
        """
        method to plot graph of resulting function and learning curve.
        """
        # first plot:
        # scatter plot of data
        x1 = list(self.data.keys())
        y1 = list(self.data.values())
        plt.scatter(x1, y1, label = "data")
        # plot function
        min_x, max_x = min(x1), max(x1)
        x2, y2 = [], []
        for p in range(PLOT_POINTS):
            x2 += [min_x + (max_x - min_x) * p / (PLOT_POINTS - 1)]
            y2 += [self.function(x2[-1], self.final_parameters)]
        plt.plot(x2, y2, label = "function")
        # axis and title
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"function fit to data: E = {self.errors[-1]:.3f}\n" + ", ".join([f"p{i} = {self.final_parameters[i]:.3f}" for i in range(self.parameter_count)]))
        plt.show()

        # second plot:
        # plot learning curve
        x3, y3 = [], []
        for p in range(PLOT_POINTS):
            # get iteration number
            i = int(p * self.max_iterations // (PLOT_POINTS - 1))
            # set iteration count and error
            x3 += [i + 1]
            y3 += [self.errors[i]]
        # plot learning curve
        plt.plot(x3, y3, label = "learning curve")
        # axis and title
        plt.xlabel("iteration")
        plt.ylabel("error")
        plt.title("learning curve")
        plt.show()
    
"""
Mathmatical model:
function f depends of x and parameters p_0, p_1, ..., p_n

each iteration parameters are updated by:
p_i = p_i - learning_rate * dE/dp_i
where dE/dp_i is derivative of error function with respect to parameter p_i

the error function E is defined as:
E = sum of (f(x) - y)**2    [for all x, y in data]
therefore the derivative of E with respect to p_i is:
dE/dp_i = dE/df * df/dp_i
        = 2 * sum(f(x) - y) * df/dp_i
"""