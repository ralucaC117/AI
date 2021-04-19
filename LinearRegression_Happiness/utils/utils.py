import csv
import matplotlib.pyplot as plt


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


def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def plotLinearity(inputs, outputs):
    plt.plot(inputs, outputs, 'ro')
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.title('GDP capita vs. happiness')
    plt.show()
