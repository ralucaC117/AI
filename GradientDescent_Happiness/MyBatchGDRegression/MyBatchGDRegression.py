import random


class MyBatchGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # lr = 0.01 epochs = 100 for univariate best results with tool scaler
    def fit(self, x, y, learningRate=0.1, noEpochs=1000):
        self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]
        # self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]

        aux = x.copy()
        for epoch in range(noEpochs):
            random.shuffle(aux)
            errors = []
            for i in range(len(x)):
                ycomputed = self.eval(x[i])
                crtError = ycomputed - y[i]
                errors.append(crtError)
            mean_errors = sum(err for err in errors) / len(errors)
            for i in range(len(x)):
                for j in range(0, len(x[0])):
                    self.coef_[j] = self.coef_[j] - learningRate * mean_errors * x[i][j]
            self.coef_[len(x[0])] = self.coef_[len(x[0])] - learningRate * mean_errors
            self.intercept_ = self.coef_[len(x[0])]
        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi):
        yi = self.intercept_
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi

    def predict(self, x):
        yComputed = [self.eval(xi) for xi in x]
        return yComputed
