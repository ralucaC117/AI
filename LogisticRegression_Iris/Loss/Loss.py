import numpy as np


def crossEntropyLoss(computedOutputs, realLabels):
    loss_sum = 0
    for i in range(len(realLabels)):
        j = realLabels[i]
        loss_sum += (-1 * np.log(computedOutputs[i][j]))
    loss = loss_sum / len(realLabels)
    return loss

# realLabels = [0, 1, 0, 1, 1, 2]
# computedOutputs = [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.5, 0.25, 0.25], [0.1, 0.1, 0.8], [0.2, 0.7, 0.1],
#                    [0.2, 0.2, 0.6]]
#
# print(str(log_loss(realLabels, computedOutputs)))
# print(str(crossEntropyLoss(computedOutputs, realLabels)))
