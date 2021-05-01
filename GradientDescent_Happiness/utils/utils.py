import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d
from math import sqrt


def loadData(fileName, inputVariabNames, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1

    selectedVariables = [dataNames.index(inputVariabName) for inputVariabName in inputVariabNames]
    inputs = [[] for i in range(len(inputVariabNames))]

    for i in range(len(inputVariabNames)):
        inputs[i] = [float(data[j][selectedVariables[i]]) for j in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


def split_data(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs[0]))]

    trainSample = np.random.choice(indexes, int(0.8 * len(inputs[0])), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]

    trainInputs = [[] for _ in range(len(inputs))]
    for j in trainSample:
        for i in range(len(inputs)):
            trainInputs[i].append(inputs[i][j])
    trainOutputs = [outputs[i] for i in trainSample]

    validationInputs = [[] for _ in range(len(inputs))]
    for j in validationSample:
        for i in range(len(inputs)):
            validationInputs[i].append(inputs[i][j])
    validationOutputs = [outputs[i] for i in validationSample]

    xx = [[] for i in range(len(trainInputs[0]))]
    for j in range(len(trainInputs[0])):
        for i in range(len(trainInputs)):
            xx[j].append(trainInputs[i][j])

    yy = [[] for i in range(len(validationInputs[0]))]
    for j in range(len(validationInputs[0])):
        for i in range(len(validationInputs)):
            yy[j].append(validationInputs[i][j])

    # xx, yy = normalisation(xx, yy)
    # trainOutputs, validationOutputs = normalisation(trainOutputs, validationOutputs)

    xx, yy = myNormalisation(xx, yy)
    trainOutputs, validationOutputs = myNormalisation(trainOutputs, validationOutputs)

    # feature1train = [ex[0] for ex in xx]
    # feature2train = [ex[1] for ex in xx]
    #
    # feature1test = [ex[0] for ex in yy]
    # feature2test = [ex[1] for ex in yy]

    # ax = plt.axes(projection='3d')
    # plt.scatter(feature1train, feature2train, trainOutputs, c="y", marker="^", label="train data")
    # plt.scatter(feature1test, feature2test, validationOutputs, c="r", marker=">", label="test data")
    # ax.set_xlabel("capita")
    # ax.set_ylabel("freedom")
    # ax.set_zlabel("happiness")
    # plt.legend()
    # plt.show()

    return xx, trainOutputs, yy, validationOutputs


def plot_learnt_model_univariate(trainInputs, trainOutputs, intercept_, coef_):
    xref = trainInputs
    yref = [None for _ in range(len(trainInputs))]
    for j in range(len(trainInputs)):
        yref[j] = intercept_
        for i in range(len(trainInputs[0])):
            yref[j] += xref[j][i] * coef_[i]

    x = [row[0] for row in trainInputs]
    plt.plot(x, trainOutputs, 'ro',
             label='training data')  # train data are plotted by red and circle sign
    plt.plot(x, yref, 'b-', label='learnt model')  # model is plotted by a blue line
    plt.title('train data and the learnt model')
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def plot_computed_outputs_univariate(validationInputs, validationOutputs, computedOutputs):
    plt.plot(validationInputs, computedOutputs, 'yo',
             label='computed test data')
    plt.plot(validationInputs, validationOutputs, 'g^', label='real test data')
    plt.title('computed test and real test data')
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


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


def myNormalisation(trainData, testData):
    if not isinstance(trainData[0], list):
        meanValue = sum(trainData) / len(trainData)
        stdDevValue = (1 / len(trainData) * sum([(feat - meanValue) ** 2 for feat in trainData])) ** 0.5
        normalisedTrainData = [(feat - meanValue) / stdDevValue for feat in trainData]
        normalisedTestData = [(feat - meanValue) / stdDevValue for feat in testData]
        return normalisedTrainData, normalisedTestData

    sumFeature1 = 0.0
    sumFeature2 = 0.0
    isBivariate = False
    for features in trainData:
        sumFeature1 += features[0]
        if len(features) == 2:
            sumFeature2 += features[1]
            isBivariate = True

    meanFeature1 = sumFeature1 / len(trainData)
    if len(trainData[0]) == 2:
        meanFeature2 = sumFeature2 / len(trainData)

    stdDevFeature1 = (1 / len(trainData) * sum([(features[0] - meanFeature1) ** 2 for features in trainData])) ** 0.5
    if isBivariate:
        stdDevFeature2 = (1 / len(trainData) * sum([(features[1] - meanFeature2) ** 2 for features in trainData])) ** 0.5

    if isBivariate:
        normalisedTrainData = [
            [(features[0] - meanFeature1) / stdDevFeature1, (features[1] - meanFeature2) / stdDevFeature2] for features in
            trainData]
        normalisedTestData = [
            [(features[0] - meanFeature1) / stdDevFeature1, (features[1] - meanFeature2) / stdDevFeature2] for features in
            testData]
    else:
        normalisedTrainData = [
            [(features[0] - meanFeature1) / stdDevFeature1] for features
            in
            trainData]
        normalisedTestData = [
            [(features[0] - meanFeature1) / stdDevFeature1] for features
            in
            testData]
    return normalisedTrainData, normalisedTestData


def plot3Ddata(x1Train, x2Train, yTrain, x1Model=None, x2Model=None, yModel=None, x1Test=None, x2Test=None, yTest=None,
               title=None):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')
    if x1Train is not None:
        plt.scatter(x1Train, x2Train, yTrain, c="r", marker="o", label="train data")
    if x1Model is not None:
        plt.scatter(x1Model, x2Model, yModel, c="b", marker="_", label='learnt model')
    if x1Test is not None:
        plt.scatter(x1Test, x2Test, yTest, c="g", marker=">", label="test data")
    # plt.title(title)
    ax.set_xlabel("capita")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.legend()
    plt.show()


def plot_learnt_model_bivariate(inputs, trainInputs, trainOutputs, intercept_, coef_):
    noOfPoints = 50
    xref1 = []
    val = min(inputs[0])
    step1 = (max(inputs[0]) - min(inputs[0])) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1

    xref2 = []
    val = min(inputs[1])
    step2 = (max(inputs[1]) - min(inputs[1])) / noOfPoints
    for _ in range(1, noOfPoints):
        aux = val
        for _ in range(1, noOfPoints):
            xref2.append(aux)
            aux += step2
    yref = [intercept_ + coef_[0] * el1 + coef_[1] * el2 for el1, el2 in zip(xref1, xref2)]
    feature1train = [ex[0] for ex in trainInputs]
    feature2train = [ex[1] for ex in trainInputs]

    ax = plt.axes(projection='3d')
    plt.scatter(feature1train, feature2train, trainOutputs, c="b", marker="^", label="train data")
    plt.scatter(xref1, xref2, yref, c="r", marker=">", label="test data")
    ax.set_xlabel("capita")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.legend()
    plt.show()


def plot_computed_outputs_bivariate(validationInputs, validationOutputs, computedOutputs):
    feature1validation = [ex[0] for ex in validationInputs]
    feature2validation = [ex[1] for ex in validationInputs]

    ax = plt.axes(projection='3d')
    ax.scatter(feature1validation, feature2validation, computedOutputs, c="y", marker="^", label="computed test data")
    ax.scatter(feature1validation, feature2validation, validationOutputs, c="green", marker="_", label="real test data")
    ax.set_xlabel("capita")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.legend()
    plt.show()
