import random
from ACO_implementation import Ant


class ACO:
    def __init__(self, problParam):
        self.__problParam = problParam
        self.__noOfAnts = problParam['NoOfAnts']
        self.__pheromoneTrailMatrix = []
        self.__Ants = []

    def initialization(self):
        self.__pheromoneTrailMatrix = [[1 for i in range(self.__problParam['NoOfNodes'])] for j in
                                       range(self.__problParam['NoOfNodes'])]

    def initializationOfAnts(self):
        self.__Ants = []
        for _ in range(self.__problParam['NoOfAnts']):
            startingNode = random.randint(0, self.__problParam['NoOfNodes'] - 1)
            ant = Ant.Ant(startingNode, self.__problParam['costsMatrix'])
            self.__Ants.append(ant)

    def computeValue(self, i, j):
        return pow(self.__pheromoneTrailMatrix[i][j], self.__problParam['alpha']) * pow(
            1/self.__problParam['costsMatrix'][i][j],
            self.__problParam['beta'])

    def chooseBest(self, ant):
        i = ant.visited[-1]
        bestChoice = float('-inf')
        bestNode = -1
        for j in range(self.__problParam['NoOfNodes']):
            if j not in ant.visited:
                value = self.computeValue(i, j)
                if value > bestChoice:
                    bestChoice = value
                    bestNode = j
        return bestNode

    def rouletteSelection(self, ant):
        i = ant.visited[-1]
        probabilities = [0] * self.__problParam['NoOfNodes']
        s = 0
        for j in range(self.__problParam['NoOfNodes']):
            if j not in ant.visited and j != i:
                probabilities[j] = (self.computeValue(i, j))
                s += probabilities[j]
        for j in range(self.__problParam['NoOfNodes']):
            if j not in ant.visited:
                if s != 0:
                    probabilities[j] /= s
        roulette = []
        for j in range(self.__problParam['NoOfNodes']):
            if j not in ant.visited:
                roulette.append([probabilities[j], j])
        for k in range(1, len(roulette)):
            roulette[k][0] += roulette[k - 1][0]
        p = random.uniform(0, max(probabilities))
        begin = 0
        for item in roulette:
            end = item[0]
            if begin < p <= end:
                return item[1]
            begin = item[0]

    def chooseNextNode(self, ant):
        q = random.uniform(0, 1)
        if q <= self.__problParam['probabilityToChooseBest']:
            return self.chooseBest(ant)
        else:
            return self.rouletteSelection(ant)

    def updatePheromone(self, bestAnt):
        for i in range(self.__problParam['NoOfNodes']):
            for j in range(self.__problParam['NoOfNodes']):
                self.__pheromoneTrailMatrix[i][j] = (1 - self.__problParam['trailEvaporationCoefficient']) * \
                                                    self.__pheromoneTrailMatrix[i][j]
        bestTrail = bestAnt.visited
        for node in range(len(bestAnt.visited) - 1):
            self.__pheromoneTrailMatrix[bestTrail[node]][bestTrail[node + 1]] += self.__problParam[
                                                                          'trailEvaporationCoefficient'] / bestAnt.trailLength
        for i in range(self.__problParam['NoOfNodes']):
            for j in range(self.__problParam['NoOfNodes']):
                if self.__pheromoneTrailMatrix[i][j] < self.__problParam['min']:
                    self.__pheromoneTrailMatrix[i][j] = self.__problParam['min']
                if self.__pheromoneTrailMatrix[i][j] > self.__problParam['max']:
                    self.__pheromoneTrailMatrix[i][j] = self.__problParam['max']

    def run(self):
        for _ in range(self.__problParam['NoOfNodes'] - 1):
            for ant in self.__Ants:
                ant.addNodeToVisited(self.chooseNextNode(ant))
        self.updatePheromone(self.bestAnt())

    def bestAnt(self):
        minLength = float('inf')
        bestAnt = None
        for ant in self.__Ants:
            if ant.trailLength < minLength:
                minLength = ant.trailLength
                bestAnt = ant
        return bestAnt

