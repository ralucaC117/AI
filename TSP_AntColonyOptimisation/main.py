from ACO_implementation.ACO_impl import ACO
from utils.dynamization import dynamize
from utils.distance import euclideanDistance

f = open('data/hard')
input = [[int(num) for num in line.split()] for line in f]
n = int(input[0][0])
costsMatrix = input[1:n + 1]


best = float('inf')
problParam = {'NoOfNodes': n, 'costsMatrix': costsMatrix, 'noOfCycles': 20,
              'alpha': 1, 'beta': 5, 'trailEvaporationCoefficient': 0.5,
              'NoOfAnts': n, 'probabilityToChooseBest': 0.3,
              'min': 1, 'max': 2}

aco = ACO(problParam)
aco.initialization()

for i in range(problParam['noOfCycles']):
    aco.initializationOfAnts()
    aco.run()
    print("Cycle " + str(i + 1))
    print(str(aco.bestAnt().visited) + " " + str(aco.bestAnt().trailLength))
    if aco.bestAnt().trailLength < best:
        best = aco.bestAnt().trailLength
    dynamize(costsMatrix, n)

print("\nbest trail length found: " + str(best))
