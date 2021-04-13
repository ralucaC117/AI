import random

def dynamize(matrix, n):
    modificationFactor = random.randint(1, 5)
    for i in range(n):
        for j in range(n):
            probability = random.uniform(0, 1)
            if probability <= 0.2:
                matrix[i][j] += modificationFactor
