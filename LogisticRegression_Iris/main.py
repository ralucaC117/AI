from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from utils.utils import *
from MyLogisticRegression.MyLogisticRegression import MyLogisticRegression
from CrossValidation.CrossValidation import CrossValidation
from sklearn.metrics import log_loss
from Loss.Loss import *

inputs, outputs = load_data()
trainInputs, testInputs, trainOutputs, testOutputs = split_data(inputs, outputs)
trainInputs, testInputs = normalisation(trainInputs, testInputs)

# -------------------------------------tool logistic regression------------------------------------
print("--------------------------------Using the sklearn logistic regressor:-----------------------")
classifier = linear_model.LogisticRegression()
classifier.fit(trainInputs, trainOutputs)
computedTestOutputs = classifier.predict(testInputs)
error = 1 - accuracy_score(testOutputs, computedTestOutputs)
print(classifier.intercept_, classifier.coef_)
print("classification error (tool): ", error)
print("accuracy score: ", accuracy_score(testOutputs, computedTestOutputs))

# -------------------------------------my logistic regression---------------------------------------
print("--------------------------------Using my logistic regressor:-----------------------------------")

classifierSetosa = MyLogisticRegression()
trainOutputsSetosa = [1 if output == 0 else 0 for output in trainOutputs]
classifierSetosa.fit(trainInputs, trainOutputsSetosa)
computedTestValuesSetosa = classifierSetosa.predict(testInputs)
print("Intercept1: " + str(classifierSetosa.intercept_))
print("Coef1: " + str(classifierSetosa.coef_))

classifierVersicolour = MyLogisticRegression()
trainOutputsVersicolour = [1 if output == 1 else 0 for output in trainOutputs]
classifierVersicolour.fit(trainInputs, trainOutputsVersicolour)
computedTestValuesVersicolor = classifierVersicolour.predict(testInputs)
print("Intercept2: " + str(classifierVersicolour.intercept_))
print("Coef2: " + str(classifierVersicolour.coef_))

classifierVirginica = MyLogisticRegression()
trainOutputsVirginica = [1 if output == 2 else 0 for output in trainOutputs]
classifierVirginica.fit(trainInputs, trainOutputsVirginica)
computedTestValuesVirginica = classifierVirginica.predict(testInputs)
print("Intercept3: " + str(classifierVirginica.intercept_))
print("Coef3: " + str(classifierVirginica.coef_))

computedTestValues = [
    [computedTestValuesSetosa[i], computedTestValuesVersicolor[i], computedTestValuesVirginica[i]]
    for i in range(len(testOutputs))]
computedTestOutputs = [np.argmax(computedTestValue) for computedTestValue in computedTestValues]

# ---------------------------------------------Loss functions-----------------------------------------
error = 1 - accuracy_score(testOutputs, computedTestOutputs)
print("classification error (tool): ", error)
print("accuracy score: ", accuracy_score(testOutputs, computedTestOutputs))

log_loss_sklearn = log_loss(testOutputs, computedTestValues)
my_loss = crossEntropyLoss(computedTestValues, testOutputs)
print("Manual log loss: " + str(my_loss))
print("Sklearn log loss: " + str(log_loss_sklearn))

# -------------------------------------Cross validation---------------------------------------
print("----------------------------------------Cross validation----------------------------------------------")
crossValidator = CrossValidation(inputs, outputs)
print("Overall accuracy: " + str(crossValidator.eval()))

# theta -> 0 => creste numarul de FP => scade precizia(tp/tp+fp)
# theta -> 1 => creste numarul de FN => scade rapelul(tp/tp+fn)
