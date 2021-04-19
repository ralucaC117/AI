from data.classificationData import realLabels, computedLabels, labelNames, computedOutputs
from eval.classification import Classification
import numpy as np

# classification = Classification(labelNames, realLabels, computedLabels, None)
# for probabilities:
classification = Classification(labelNames, realLabels, None, computedOutputs)

#classification.run()
#print(classification.binaryLoss())
#print(classification.multiClassLoss())
#print(classification.multiLabelLoss())
