from MyStochasticGDRegression.MyStochasticGDRegression import *


class MyMultiOutputRegressor:
    def __init__(self):
        self.estimators = []

    def fit(self, x, y):
        for i in range(len(y[0])):
            sRegressor = MyStochasticGDRegression()
            sRegressor.fit(x, [row[i] for row in y])
            self.estimators.append([sRegressor.intercept_, sRegressor.coef_])

    def predict(self, x):
        computedOutputs = [[] for _ in range(len(x))]

        for estimator in self.estimators:
            for i in range(len(x)):
                computedOutputs[i].append(self.eval(x[i], estimator))
        return computedOutputs

    def eval(self, xi, estimator):
        yi = estimator[0]
        for j in range(len(xi)):
            yi += estimator[1][j] * xi[j]
        return yi



