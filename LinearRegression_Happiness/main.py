import os
from utils.utils import *
from tool.linearRegression import *
from nolibs.myUnivariateRegression import *

crtDir = os.getcwd()
filePath = os.path.join(crtDir, 'data', 'world-happiness-report-2017.csv')

inputs, outputs = loadData(filePath, ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')
# plotDataHistogram(inputs, 'capita GDP')
# plotDataHistogram(outputs, 'Happiness score')
# plotLinearity(inputs, outputs)

regressor = LinearRegression(inputs, outputs)
regressor.learn()
# regressor.plot_learnt_model_univariate()
# regressor.predict_univariate()
regressor.plot_learnt_model_bivariate()
regressor.predict_bivariate()

# regressor = MyLinearUnivariateRegression(inputs, outputs)
# regressor.learn()
# regressor.plot_learnt_model()
# regressor.plot_predicted()
