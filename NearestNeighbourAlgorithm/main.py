from myGraph import MyGraph

# read from file
f = open("hard.txt")
input = [[int(num) for num in line.split(',')] for line in f]

# assign the values to variables
n = int(input[0][0])
matrix = input[1:n + 1]
source = int(input[-2][0])
destination = int(input[-1][0])

# run the algorithm
myGraph = MyGraph(n, matrix)
sol1 = myGraph.nearestNeighbourPath()
sol2 = myGraph.nearestNeighbourPathWithSourceAndDestination(source, destination)


# write solution to file
with open("solution.txt", "w") as file:
    file.write(str(sol1[0]) + "\n")
    file.write(str(sol1[1]) + "\n")
    file.write(str(sol2[0]) + "\n")
    file.write(str(sol2[1]) + "\n")
    file.write(str(sol2[2]) + "\n")

print("Check the solution.txt file")
