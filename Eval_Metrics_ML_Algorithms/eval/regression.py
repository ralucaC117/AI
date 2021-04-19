from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Regression:
    def __init__(self, realOutputs, computedOutputs):
        self.__realOutputs = realOutputs
        self.__computedOutputs = computedOutputs

    def predictionError(self):
        merged_real = [item for sublist in self.__realOutputs for item in sublist]
        merged_computed = [item for sublist in self.__computedOutputs for item in sublist]
        # MAE
        mae = sum(abs(r - c) for r, c in zip(merged_real, merged_computed)) / len(merged_real)
        # MSRE
        msre = sqrt(sum((r - c) ** 2 for r, c in zip(merged_real, merged_computed)) / len(merged_real))

        # MAE using sklearn
        mae_sklearn = mean_absolute_error(self.__realOutputs, self.__computedOutputs)
        # MSRE using sklearn
        msre_sklearn = sqrt(mean_squared_error(self.__realOutputs, self.__computedOutputs))
        return [{'MAE': mae, 'MSRE': msre}, {'MAE_sklearn': mae_sklearn, 'MSRE_sklearn': msre_sklearn}]

    def loss(self):
        return self.predictionError()

    def runEval(self):
        print("performance metrics for a regression problem: ")
        print("prediction error:\n" + str(self.predictionError()[0]) + "\n" + str(self.predictionError()[1]))
        print("loss:\n" + str(self.loss()[0]) + "\n" + str(self.loss()[1]))
