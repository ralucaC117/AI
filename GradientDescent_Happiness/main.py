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

inputs, outputs = loadData(filePath, ['Economy..GDP.per.Capita.'], 'Happiness.Score')
trainInputs, trainOutputs, validationInputs, validationOutputs = split_data(inputs, outputs)

# sRegressor = MyStochasticGDRegression()
# sRegressor.fit(trainInputs, trainOutputs)
# w0, w1 = sRegressor.intercept_, sRegressor.coef_
# computedValidationOutputs = sRegressor.predict(validationInputs)
# print(w0, w1)

# bRegressor = BatchGDRegression()
bRegressor = MyBatchGDRegression()
bRegressor.fit(trainInputs, trainOutputs)
w0, w1 = bRegressor.intercept_, bRegressor.coef_
computedValidationOutputs = bRegressor.predict(validationInputs)
print(w0, w1)

# univariate GD
plot_learnt_model_univariate(trainInputs, trainOutputs, w0, w1)
plot_computed_outputs_univariate(validationInputs, validationOutputs, computedValidationOutputs)

# bivariate GD
# plot_learnt_model_bivariate(inputs, trainInputs, trainOutputs, w0, w1)
# plot_computed_outputs_bivariate(validationInputs, validationOutputs, computedValidationOutputs)

error = mean_squared_error(validationOutputs, computedValidationOutputs)
print('prediction error (tool):  ', error)
