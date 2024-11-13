import numpy as np
import matplotlib.pyplot as plt
import argparse

class SimpleLinearRegression:
    def __init__(self, X: np.array, Y: np.array):
        self.__n = None
        self.__sum_X = None
        self.__sum_Y = None
        self.__sum_XY = None
        self.__sum_X_squares = None
        self.beta0 = None
        self.beta1 = None

        self.calculate_parameters(X, Y)

    def __calculate_beta1(self):
        numerator = (self.__n * self.__sum_XY) -\
            (self.__sum_X * self.__sum_Y)
        denominator = (self.__n * self.__sum_X_squares) -\
            (self.__sum_X * self.__sum_X)

        return numerator / denominator

    def __calculate_beta0(self):
        numerator = (self.__sum_X_squares * self.__sum_Y) -\
            (self.__sum_X * self.__sum_XY)
        denominator = (self.__n * self.__sum_X_squares) -\
            (self.__sum_X * self.__sum_X)

        return numerator / denominator

    def calculate_parameters(self, X: np.array, Y: np.array):
        self.__n = X.size
        self.__sum_X = np.sum(X)
        self.__sum_Y = np.sum(Y)
        self.__sum_XY = np.sum(X*Y)
        self.__sum_X_squares = np.sum(np.pow(X, 2))

        self.beta0 = self.__calculate_beta0()
        self.beta1 = self.__calculate_beta1()

    def predict(self, x: np.float64):
        return self.beta0 + (self.beta1 * x)

    def get_parameters(self):
        return self.beta0, self.beta1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("values_to_predict", help="Values separated by comma ','")
    args = parser.parse_args()

    # Data set
    advertising = np.array([1,2,3,4,5,6,7,8,9])
    sales = np.array([2,4,6,8,10,12,14,16,18])

    slr = SimpleLinearRegression(advertising, sales)
    beta0, beta1 = slr.get_parameters()
    print(f"y = {beta0:.4f} + {beta1:.4f}x")

    values_to_predict = [float(v) for v in args.values_to_predict.split(',')]
    predictions = list()
    for value in values_to_predict:
        prediction = slr.predict(value)
        predictions.append(prediction)
        print(f"For {value} the predicted value is: {prediction:.4f}")

    fig, ax = plt.subplots()

    # Plot original data set and the predictions made
    ax.scatter(advertising, sales, label="Actual Data")
    ax.scatter(np.asarray(values_to_predict), np.asarray(predictions), color="orange", label="Predictions")

    # Plot the line that best fit the original data set
    x_line = np.linspace(0, 100, 10, endpoint=True)
    y_line = np.array([slr.predict(x) for x in x_line ])
    ax.plot(x_line, y_line, color='red', label=f"y = {beta0:.4f} + {beta1:.4f}x")

    # Set labels
    ax.set_xlabel('Advertising')
    ax.set_ylabel('Sales')
    ax.set_title('Scatter Plot with Regression Line and Predictions')
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
