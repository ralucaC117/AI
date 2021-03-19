from random import randint, sample
from utils.utils import generateNewValue
import collections

class Chromosome:
    def __init__(self, problParam=None):
        self.__problParam = problParam
        self.__repres = [generateNewValue(problParam['min'], problParam['max']) for _ in range(problParam['noDim'])]
        self.__fitness = 0.0
        # randomly selecting 20% of the nodes and assign their community IDs to all of their neighbours
        noOfNodes = int(20/100*problParam['noDim'])
        nodes = sample(self.__problParam['graph'].nodes, noOfNodes)
        for node in nodes:
            for neighbour in self.__problParam['graph'].neighbors(node):
                genes = self.__repres
                genes[self.__problParam['dict_of_nodes'][neighbour]] = genes[self.__problParam['dict_of_nodes'][node]]
                self.__repres = genes
        self.__repres = self.renumbering(self.__repres)

    @property
    def repres(self):
        return self.__repres

    @property
    def fitness(self):
        return self.__fitness

    @repres.setter
    def repres(self, l=[]):
        self.__repres = l

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    # one way crossover
    def crossover(self, c):
        randPos = randint(0, len(self.__repres)-1)
        source = self.__repres
        destination = c.__repres
        newrepres = []
        communityNo = source[randPos]
        for i in range(0, len(self.__repres)):
            if source[i] == communityNo:
                newrepres.append(communityNo)
            else:
                newrepres.append(destination[i])

        offspring = Chromosome(c.__problParam)
        offspring.repres = self.renumbering(newrepres)
        return offspring


    def mutation(self):
        pos_i = randint(0, len(self.__repres) - 1)
        pos_j = randint(0, len(self.__repres) - 1)
        self.__repres[pos_j] = self.__repres[pos_i]
        self.__repres = self.renumbering(self.__repres)

    def renumbering(self, repres):
        initial_repres = repres
        communities_dict = {}
        for i in range(len(initial_repres)):
            if initial_repres[i] in communities_dict:
                communities_dict[initial_repres[i]].append(i)
            else:
                communities_dict[initial_repres[i]] = [i]

        smallest = 1
        for key in sorted(communities_dict):
                if smallest not in communities_dict:
                    communities_dict[smallest] = communities_dict[key]
                    del communities_dict[key]
                    smallest += 1
                else:
                    smallest += 1

        final_repres = [-1]*len(initial_repres)
        for key in communities_dict.keys():
            for value in communities_dict[key]:
                final_repres[value] = key

        return final_repres


    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness
