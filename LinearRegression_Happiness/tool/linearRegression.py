import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, inputs, outputs):
        self.__inputs = inputs
        self.__outputs = outputs
        self.intercept_ = 0.0
        self.coef_ = None
        self.regressor = None

    def split_data(self):
        np.random.seed(5)
        indexes = [i for i in range(len(self.__inputs[0]))]
        trainSample = np.random.choice(indexes, int(0.8 * len(self.__inputs[0])), replace=False)
        validationSample = [i for i in indexes if not i in trainSample]

        trainInputs = [[] for _ in range(len(self.__inputs))]
        for j in trainSample:
            for i in range(len(self.__inputs)):
                trainInputs[i].append(self.__inputs[i][j])
        # trainInputs = [self.__inputs[i][j] for i in range(len(self.__inputs)) for j in trainSample]
        trainOutputs = [self.__outputs[i] for i in trainSample]

        validationInputs = [[] for _ in range(len(self.__inputs))]
        for j in validationSample:
            for i in range(len(self.__inputs)):
                validationInputs[i].append(self.__inputs[i][j])

        # validationInputs = [self.__inputs[i][j] for j in validationSample for i in range(len(self.__inputs))]
        validationOutputs = [self.__outputs[i] for i in validationSample]

        return trainInputs, trainOutputs, validationInputs, validationOutputs

    def learn(self):
        trainInputs, trainOutputs = self.split_data()[0:2]
        xx = [[] for i in range(len(trainInputs[0]))]
        for j in range(len(trainInputs[0])):
            for i in range(len(trainInputs)):
                xx[j].append(trainInputs[i][j])
        regressor = linear_model.LinearRegression()
        regressor.fit(xx, trainOutputs)
        self.intercept_, self.coef_ = regressor.intercept_, regressor.coef_
        self.regressor = regressor
        print('the learnt model: ', self.intercept_, ' + ', self.coef_)

    def plot_learnt_model_univariate(self):
        trainInputs, trainOutputs = self.split_data()[0:2]
        xref = trainInputs
        yref = [None for _ in range(len(trainInputs[0]))]
        for j in range(len(trainInputs[0])):
            yref[j] = self.intercept_
            for i in range(len(trainInputs)):
                yref[j] += xref[i][j] * self.coef_[i]

        plt.plot(trainInputs[0], trainOutputs, 'ro',
                 label='training data')  # train data are plotted by red and circle sign
        plt.plot(xref[0], yref, 'b-', label='learnt model')  # model is plotted by a blue line
        plt.title('train data and the learnt model')
        plt.xlabel('GDP capita')
        plt.ylabel('happiness')
        plt.legend()
        plt.show()

    def plot_learnt_model_bivariate(self):
        trainInputs, trainOutputs = self.split_data()[0:2]
        xref = trainInputs
        yref = [None for _ in range(len(trainInputs[0]))]
        for j in range(len(trainInputs[0])):
            yref[j] = self.intercept_
            for i in range(len(trainInputs)):
                yref[j] += xref[i][j] * self.coef_[i]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(trainInputs[0], trainInputs[1], trainOutputs, c="blue", label="training data")
        ax.scatter(xref[0], xref[1], yref, c="red", label="learnt model")
        plt.xlabel('GDP capita')
        plt.ylabel('freedom')
        plt.legend()
        plt.show()

    def predict_univariate(self):
        validationInputs, validationOutputs = self.split_data()[2:]
        xx = [[] for i in range(len(validationInputs[0]))]
        for j in range(len(validationInputs[0])):
            for i in range(len(validationInputs)):
                xx[j].append(validationInputs[i][j])
        computedValidationOutputs = self.regressor.predict(xx)
        plt.plot(validationInputs[0], computedValidationOutputs, 'yo',
                 label='computed test data')
        plt.plot(validationInputs[0], validationOutputs, 'g^',
                 label='real test data')
        plt.title('computed validation and real validation data')
        plt.xlabel('GDP capita')
        plt.ylabel('happiness')
        plt.legend()
        plt.show()

        error = mean_squared_error(validationOutputs, computedValidationOutputs)
        print('prediction error (tool):  ', error)


    def predict_bivariate(self):
        validationInputs, validationOutputs = self.split_data()[2:]
        xx = [[] for i in range(len(validationInputs[0]))]
        for j in range(len(validationInputs[0])):
            for i in range(len(validationInputs)):
                xx[j].append(validationInputs[i][j])
        computedValidationOutputs = self.regressor.predict(xx)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(validationInputs[0], validationInputs[1], computedValidationOutputs, c="y", marker="^", label="computed test data", alpha=1)
        ax.scatter(validationInputs[0], validationInputs[1], validationOutputs, c="green", label="real test data", alpha=1)
        plt.xlabel('GDP capita')
        plt.ylabel('freedom')
        plt.legend()
        plt.show()

        error = mean_squared_error(validationOutputs, computedValidationOutputs)
        print('prediction error (tool):  ', error)
