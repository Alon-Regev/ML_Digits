from GradientDescent import GradientDescent

def main():
    # create data
    data = {
        1: 0.5,
        2: 0.75,
        3: 1.2,
        4: 1.5,
        5: 1.7,
        6: 2.0,
        7: 1.9,
        8: 2.2,
        9: 2.7,
        10: 2.9,
        12.5: 3.5,
    }
    # create linear function
    # y = ax + b
    f = lambda x, params: params[0] * x + params[1]
    # create derivatives
    derivatives = [
        # df/da = x
        lambda x, params: x,
        # df/db = 1
        lambda x, params: 1
    ]
    # create gradient descent object
    gd = GradientDescent(
        learning_rate=0.0001, 
        max_iterations=1000, 
        parameter_count=2, 
        data=data, 
        function=f, 
        derivatives=derivatives
    )
    # fit data to function
    gd.fit()
    # show plot
    gd.plot()

if __name__ == "__main__":
    main()