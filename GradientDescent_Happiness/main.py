import os
from utils.utils import *
from MyStochasticGDRegression.MyStochasticGDRegression import *
from MyBatchGDRegression.MyBatchGDRegression import *
from BatchGDRegression.BatchGDRegression import *
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings("ignore")

crtDir = os.getcwd()
filePath = os.path.join(crtDir, 'data', 'world-happiness-report-2017.csv')

inputs, outputs = loadData(filePath, ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')
trainInputs, trainOutputs, validationInputs, validationOutputs = split_data(inputs, outputs)

# ------------Stochastic Regressor--------------------------
# sRegressor = MyStochasticGDRegression()
# sRegressor.fit(trainInputs, trainOutputs)
# w0, w1 = sRegressor.intercept_, sRegressor.coef_
# computedValidationOutputs = sRegressor.predict(validationInputs)
# print(w0, w1)


bRegressor = BatchGDRegression()
mybRegressor = MyBatchGDRegression()

bRegressor.fit(trainInputs, trainOutputs)
w0, w1 = bRegressor.intercept_, bRegressor.coef_
computedValidationOutputs = bRegressor.predict(validationInputs)
print("-------------------Tool:----------------------------------")
print(w0, w1)
error = mean_squared_error(validationOutputs, computedValidationOutputs)
print('prediction error', error)


mybRegressor.fit(trainInputs, trainOutputs)
w0, w1 = mybRegressor.intercept_, mybRegressor.coef_
computedValidationOutputs = mybRegressor.predict(validationInputs)
print("--------------------My regressor:------------------------- ")
print(w0, w1)
error = mean_squared_error(validationOutputs, computedValidationOutputs)
print('prediction error', error)

# ------------------------plot univariate GD----------------------------
# plot_learnt_model_univariate(trainInputs, trainOutputs, w0, w1)
# plot_computed_outputs_univariate(validationInputs, validationOutputs, computedValidationOutputs)

# ------------------------plot bivariate GD-----------------------------
plot_learnt_model_bivariate(inputs, trainInputs, trainOutputs, w0, w1)
plot_computed_outputs_bivariate(validationInputs, validationOutputs, computedValidationOutputs)


