from GeneticAlgorithm import GA
from utils.cost import fitness_function
from Chromosome import Chromosome
import math


def euclideanDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


# read from file
# f = open('data/medium')
# input = [[int(num) for num in line.split()] for line in f]
# n = int(input[0][0])
# costs_matrix = input[1:n + 1]


# read coordinates from file
f = open('data/hard')
input = [[int(num) for num in line.split()] for line in f]
n = int(input[0][0])
init_matrix = input[1:n + 1]
costs_matrix = [[0 for i in range(n)] for j in range(n)]
for i in range(n):
    for j in range(n):
        if i != j:
            costs_matrix[i][j] = euclideanDistance(init_matrix[i][1], init_matrix[i][2], init_matrix[j][1], init_matrix[j][2])


# set params for GA
gaParam = {'popSize': 15, 'noGen': 100}
problParam = {'costs_matrix': costs_matrix, 'function': fitness_function, 'noOfNodes': n}

# initialise GA
ga = GA(gaParam, problParam)
ga.initialisation()
ga.evaluation()
best_overall_chromosome = Chromosome(problParam)
best_overall_chromosome.fitness = 100000000

# iterate through generations
with open("solution.txt", "w") as file:
    for generation in range(gaParam['noGen']):
        best_chromosome = ga.bestChromosome()
        # print(str(ga.population))
        print(str(generation) + " Best Chromosome: " + str(best_chromosome))
        file.write(str(generation) + " Best Chromosome: " + str(best_chromosome) + "\n")
        if best_chromosome.fitness < best_overall_chromosome.fitness:
            best_overall_chromosome = best_chromosome
        ga.oneGenerationElitism()
    print("Best overall chromosome: " + str(best_overall_chromosome))
    file.write("Best overall chromosome: " + str(best_overall_chromosome))
    file.close()
