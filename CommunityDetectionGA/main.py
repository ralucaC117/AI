import networkx as nx
from service.CommunityDetection import modularity
from service.GeneticAlgorithm import GA
from utils import graphUtils

#graph = nx.read_gml('data/dolphins.gml')
#graph = nx.read_gml('data/football.gml')
#graph = nx.read_gml('data/karate.gml', label='id')
# graph = nx.read_gml('data/krebs.gml', label='id')
graph = nx.read_gml('data/adjnoun.gml')
# graph = nx.read_gml('data/lesmis.gml')

dict_of_nodes = {}
for index, node in enumerate(graph.nodes):
    dict_of_nodes[node] = index

gaParam = {'popSize': 100, 'noGen': 50}
problParam = {'graph': graph, 'min': 1, 'max': len(graph.nodes) - 1, 'function': modularity, 'noDim': len(graph.nodes),
              'mat': nx.to_numpy_matrix(graph), 'noNodes': len(graph.nodes), 'degrees': graph.degree,
              'noEdges': len(graph.edges), 'dict_of_nodes': dict_of_nodes}

ga = GA(gaParam, problParam)
ga.initialisation()
ga.evaluation()

allbest = []
last = []
bestestFitness = 0.0
bestestNoOfCommunities = 0
bestesCRepr = []

allFitnessValues = []
allCommunitiesNumbers = []

for g in range(gaParam['noGen']):
    bestChromosome = ga.bestChromosome()
    # compute the number of communities
    communities_dict = {}
    for i in range(len(bestChromosome.repres)):
        if bestChromosome.repres[i] in communities_dict:
            communities_dict[bestChromosome.repres[i]].append(i)
        else:
            communities_dict[bestChromosome.repres[i]] = [i]
    allCommunitiesNumbers.append(len(communities_dict))
    allFitnessValues.append(bestChromosome.fitness)
    # store the values associated with the best chromosome
    if bestChromosome.fitness > bestestFitness:
        bestestFitness = bestChromosome.fitness
        bestesCRepr = bestChromosome.repres
        bestestNoOfCommunities = len(communities_dict)
    ga.oneGenerationElitism()

# for chromosome in allbest:
#     graphUtils.drawNetwork(graph, chromosome)

print("Numarul de comunitati ale celui mai bun cromozom: " + str(bestestNoOfCommunities))
print("Fitnessul celui mai bun cromozom: " + str(bestestFitness))
for i in range(0, len(bestesCRepr)):
    print(str(i) + ": " + str(bestesCRepr[i]))

print("Evolutia fitnessului celui mai bun cromozom: ")
print(allFitnessValues)

print("Evolutia numarului de comunitati de-a lungul generatiilor: ")
print(allCommunitiesNumbers)

graphUtils.drawNetwork(graph, bestesCRepr)


with open("solution.txt", "w") as file:
    file.write("Numarul de comunitati ale celui mai bun cromozom: " + str(bestestNoOfCommunities) + "\n")
    file.write("Fitnessul celui mai bun cromozom: " + str(bestestFitness) + "\n")
    file.write("Evolutia fitnessului celui mai bun cromozom: " + str(allFitnessValues) + "\n")
    file.write("Evolutia numarului de comunitati de-a lungul generatiilor: " + str(allCommunitiesNumbers) + "\n")
    file.write("Apartenenta indivizilor la comunitati: " + "\n")
    for i in range(0, len(bestesCRepr)):
        file.write(str(i) + ": " + str(bestesCRepr[i]))
    file.close()

