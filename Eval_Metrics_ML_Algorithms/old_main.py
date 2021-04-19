from eval.regression import Regression
from eval.classification import Classification

# multi target regression:
# realOutputs = [[7.4, 0.7, 0], [7.8, 0.88, 0], [11.2, 0.28, 0.56]]
# computedOutputs = [[7.0, 0.98, 1], [6.2, 1.89, 0], [11.2, 0.20, 0.56]]

# realOutputs0 = [3, 9.5, 4, 5.1, 6, 7.2, 2, 1]
# computedOutputs0 = [2, 7, 4.5, 6, 3, 8, 3, 1.2]
# realOutputs = [[el] for el in realOutputs0]
# computedOutputs = [[el] for el in computedOutputs0]

# print("multi target regression problem")
# regressionProblem = Regression(realOutputs, computedOutputs)
# print("prediction error: " + str(regressionProblem.predictionError()))
# print("loss: " + str(regressionProblem.loss()))

# multi class classification
# labelNames = ['apple', 'pear', 'peach']
# realLabels = ['peach', 'apple', 'pear', 'apple', 'pear', 'peach']
# computedLabels = ['apple', 'apple', 'pear', 'pear', 'pear', 'peach']
# computedOutputs = [[0.7, 0.1, 0.2], [0.5, 0.2, 0.3], [0.3, 0.4, 0.3], [0.1, 0.8, 0.1], [0.1, 0.7, 0.2], [0.4, 0.1, 0.5]]

# realLabels = ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
# computedLabels = ['spam', 'ham', 'ham', 'spam', 'spam', 'ham']
# labelNames = ['spam', 'ham']
# computedOutputs = [[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1], [0.7, 0.3], [0.4, 0.6]]

# realLabels = ['infected', 'infected', 'infected', 'infected', 'normal', 'normal', 'normal', 'normal', 'normal','normal', 'normal', 'normal', 'normal', 'normal', 'normal']
# computedLabels = ['infected', 'infected', 'normal', 'normal', 'normal', 'normal','normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'infected']
# labelNames = ['infected', 'normal']

# print("\nmulti classification problem")
# classificationProblem = ClassificationProblem(labelNames, realLabels, computedLabels, None)
# classificationProblem = Classification(labelNames, realLabels, None, computedOutputs)
#
# acc, precision, recall = classificationProblem.performance()
# print("accuracy: " + str(acc))
# print("precision: " + str(precision))
# print("recall: " + str(recall))
# print("loss: " + str(classificationProblem.loss()))

# multi label
# computed = [[1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1]]
# real = [[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1]]
# multiLabelClassification = Classification([], real, computed, None)
# print("\nmulti label loss: " + str(multiLabelClassification.multiLabelLoss()))
