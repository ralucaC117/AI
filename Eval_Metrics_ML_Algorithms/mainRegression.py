from data.regressionData import realOutputs, computedOutputs
from eval.regression import Regression

regression = Regression(realOutputs, computedOutputs)
regression.runEval()

