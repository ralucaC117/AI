import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def load_data():
    data = load_iris()
    inputs = data['data']
    featureNames = data['feature_names']
    # feature1 = [feat[featureNames.index('sepal length (cm)')] for feat in inputs]
    # feature2 = [feat[featureNames.index('sepal width (cm)')] for feat in inputs]
    # feature3 = [feat[featureNames.index('petal length (cm)')] for feat in inputs]
    # feature4 = [feat[featureNames.index('petal width (cm)')] for feat in inputs]
    inputs = [[feat[featureNames.index('sepal length (cm)')], feat[featureNames.index('sepal width (cm)')],
               feat[featureNames.index('petal length (cm)')], feat[featureNames.index('petal width (cm)')]] for feat in
              inputs]
    outputs = data['target']
    outputNames = data['target_names']
    return inputs, outputs


def split_data(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, testInputs, trainOutputs, testOutputs


def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
    return normalisedTrainData, normalisedTestData


def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x)
    plt.title('Histogram of ' + variableName)
    plt.show()

# plotDataHistogram(feature1, 'sepal length (cm)')
# plotDataHistogram(feature2, 'sepal width (cm)')
# plotDataHistogram(feature3, 'petal length (cm)')
# plotDataHistogram(feature4, 'petal width (cm)')
# plotDataHistogram(outputs, 'iris class')
