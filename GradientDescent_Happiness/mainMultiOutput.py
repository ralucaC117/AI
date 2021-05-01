from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from MyMultiOutputRegressor.MultiOutputRegressor import *

inputs, outputs = load_linnerud(return_X_y=True)
trainInputs, validationInputs, trainOutputs, validationOutputs = train_test_split(inputs, outputs, test_size=0.20,
                                                                                  random_state=1)

scaler = StandardScaler()
scaler.fit(trainInputs)
trainInputs = scaler.transform(trainInputs)
validationInputs = scaler.transform(validationInputs)

scaler.fit(trainOutputs)
trainOutputs = scaler.transform(trainOutputs)
validationOutputs = scaler.transform(validationOutputs)

print("------------------------sklearn multioutput regressor----------------")
model = MultiOutputRegressor(Ridge(random_state=1)).fit(trainInputs, trainOutputs)
predictedOutputs = model.predict(validationInputs)
error = mean_squared_error(validationOutputs, predictedOutputs)

print(model.estimators_[0].intercept_, model.estimators_[0].coef_)
print(model.estimators_[1].intercept_, model.estimators_[1].coef_)
print(model.estimators_[2].intercept_, model.estimators_[2].coef_)
print('prediction error', error)

print("------------------------my multioutput regressor--------------------")

multiOutputRegressor = MyMultiOutputRegressor()
multiOutputRegressor.fit(trainInputs, trainOutputs)
for estimator in multiOutputRegressor.estimators:
    print(estimator)
predictedOutputs = multiOutputRegressor.predict(validationInputs)
error = mean_squared_error(validationOutputs, predictedOutputs)
print('prediction error', error)
