from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, log_loss, hamming_loss
import numpy as np
from math import log


class Classification:
    def __init__(self, labelNames, realLabels, computedLabels=None, computedOutputs=None):
        self.__realLabels = realLabels
        self.__computedLabels = computedLabels
        self.__labelNames = labelNames
        self.__computedOutputs = computedOutputs

    def performanceMetrics(self):
        if self.__computedOutputs is not None:
            self.__computedLabels = [self.__labelNames[np.argmax(p)] for p in self.__computedOutputs]
        acc = accuracy_score(self.__realLabels, self.__computedLabels)
        precision = precision_score(self.__realLabels, self.__computedLabels, average=None, labels=self.__labelNames)
        recall = recall_score(self.__realLabels, self.__computedLabels, average=None, labels=self.__labelNames)
        return {'accuracy': acc, 'precision': precision, 'recall': recall}

    def run(self):
        print("performance metrics for a classification problem:\n")
        print(self.performanceMetrics())

    # binary cross entropy
    def binaryLoss(self):
        realLabels = []
        for label in self.__realLabels:
            if label == 'spam':
                realLabels.append(0)
            else:
                realLabels.append(1)
        loss_sum = 0
        for i in range(len(self.__computedOutputs)):
            if realLabels[i] == 0:
                loss_sum += np.log(self.__computedOutputs[i][1])
            else:
                loss_sum += np.log(self.__computedOutputs[i][0])

        loss = -1 * loss_sum / len(realLabels)
        loss_sklearn = log_loss(self.__realLabels, self.__computedOutputs)
        return {'loss': loss, 'loss_sklearn': loss_sklearn}

    # cross entropy
    def multiClassLoss(self):
        realLabels = []
        for label in self.__realLabels:
            if label == 'spam':
                realLabels.append([1, 0, 0])
            if label == 'ham':
                realLabels.append([0, 1, 0])
            if label == 'jam':
                realLabels.append([0, 0, 1])
        loss_sum = 0
        for i in range(len(self.__computedOutputs)):
            for j in range(len(realLabels[i])):
                if realLabels[i][j] == 1:
                    loss_sum += (-1 * np.log(self.__computedOutputs[i][j]))
        loss = loss_sum / len(realLabels)

        loss_sklearn = log_loss(realLabels, self.__computedOutputs)
        return {'loss': loss, 'loss_sklearn': loss_sklearn}

    # sum of binary cross entropies for every label
    def multiLabelLoss(self):
        loss_sum = 0
        for i in range(len(self.__computedOutputs)):
            cross_entropy = 0
            for j in range(len(self.__realLabels[i])):
                cross_entropy += (-1 * ((self.__realLabels[i][j] * np.log(self.__computedOutputs[i][j])) +
                                        ((1 - self.__realLabels[i][j]) * np.log(1 - self.__computedOutputs[i][j]))))
            loss_sum += cross_entropy
        loss = loss_sum / len(self.__realLabels)
        loss_sklearn = log_loss(self.__realLabels, self.__computedOutputs)
        return {'loss': loss, 'loss_sklearn': loss_sklearn}
