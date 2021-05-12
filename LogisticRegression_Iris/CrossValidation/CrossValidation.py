import numpy as np
from sklearn.metrics import accuracy_score

from MyLogisticRegression.MyLogisticRegression import MyLogisticRegression
from utils.utils import normalisation
import random


class CrossValidation:
    # k folded cross validation
    def __init__(self, inputs, outputs):
        self.k = 5
        self.inputs = inputs
        self.outputs = outputs
        self.folds = []

    def split(self):
        indexes = list(range(len(self.inputs)))
        random.shuffle(indexes)

        currentStart = 0
        currentEnd = len(self.inputs) // self.k
        for _ in range(self.k):
            currentFold = []
            for i in range(currentStart, currentEnd):
                currentFold.append(indexes[i])
            currentStart = currentEnd
            currentEnd += len(self.inputs) // self.k
            self.folds.append(currentFold)

    def eval(self):
        self.split()
        sum_accuracy = 0
        for k in range(self.k):
            trainInputs = []
            trainOutputs = []
            testInputs = []
            testOutputs = []

            for i in self.folds[k]:
                testInputs.append(self.inputs[i])
                testOutputs.append(self.outputs[i])

            for j in range(self.k):
                if j != k:
                    for i in self.folds[k]:
                        trainInputs.append(self.inputs[i])
                        trainOutputs.append(self.outputs[i])

            trainInputs, testInputs = normalisation(trainInputs, testInputs)

            classifierSetosa = MyLogisticRegression()
            trainOutputsSetosa = [1 if output == 0 else 0 for output in trainOutputs]
            classifierSetosa.fit(trainInputs, trainOutputsSetosa)
            computedTestOutputsValuesSetosa = classifierSetosa.predict(testInputs)

            classifierVersicolour = MyLogisticRegression()
            trainOutputsVersicolour = [1 if output == 1 else 0 for output in trainOutputs]
            classifierVersicolour.fit(trainInputs, trainOutputsVersicolour)
            computedTestOutputsValuesVersicolor = classifierVersicolour.predict(testInputs)

            classifierVirginica = MyLogisticRegression()
            trainOutputsVirginica = [1 if output == 2 else 0 for output in trainOutputs]
            classifierVirginica.fit(trainInputs, trainOutputsVirginica)
            computedTestOutputsValuesVirginica = classifierVirginica.predict(testInputs)

            computedTestValues = [[computedTestOutputsValuesSetosa[i], computedTestOutputsValuesVersicolor[i],
                                   computedTestOutputsValuesVirginica[i]] for i in range(len(testOutputs))]
            computedTestOutputs = [np.argmax(computedTestValue) for computedTestValue in computedTestValues]

            accuracyScore = accuracy_score(testOutputs, computedTestOutputs)
            print("Trial " + str(k) + ": accuracy=" + str(accuracyScore))
            sum_accuracy += accuracy_score(testOutputs, computedTestOutputs)
        return sum_accuracy / self.k
