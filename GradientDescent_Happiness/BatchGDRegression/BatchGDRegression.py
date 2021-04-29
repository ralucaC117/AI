import random
from sklearn.linear_model import SGDRegressor
import numpy as np

class BatchGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []
        self.classifier = SGDRegressor(loss="squared_loss", learning_rate="constant", eta0=0.001)

    def fit(self, x, y, noEpochs=1000):
        for epoch in range(noEpochs):
            self.classifier.partial_fit(x, y)
            self.intercept_ = self.classifier.intercept_[0]
            self.coef_ = self.classifier.coef_

    def predict(self, x):
        return self.classifier.predict(x)
