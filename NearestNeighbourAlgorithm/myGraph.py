class MyGraph():
    def __init__(self, n, costsMatrix):
        self.n = n
        self.costsMatrix = costsMatrix

    def printGraph(self):
        print(self.n)
        print(self.costsMatrix)

    # finds the nearest node based on the costs matrix
    def nearest(self, source, visited):
        minimum = 10000000;
        neighbour = source
        for i in range(0, self.n):
            if self.costsMatrix[source][i] != 0 and self.costsMatrix[source][i] <= minimum and visited[i] == "false":
                minimum = self.costsMatrix[source][i]
                neighbour = i
        return neighbour

    # returns the shortest path that starts from the first vertex and visits all of the other vertexes
    # and its length based on greedy search
    def nearestNeighbourPath(self):
        visited = ["false"] * self.n
        source = 0
        path = [source+1]
        visited[source] = "true"
        cost = 0
        for i in range(1, self.n):
            nextNeighbour = self.nearest(source, visited)
            if nextNeighbour != source:
                cost += self.costsMatrix[source][nextNeighbour]
                source = nextNeighbour
                visited[nextNeighbour] = "true"
                path.append(nextNeighbour+1)
        cost += self.costsMatrix[source][0]
        return [path, cost]

    # returns the path that starts from the source vertex and gets to the destination vertex
    # based on greedy local search, the number of vertexes that are visited and the length
    def nearestNeighbourPathWithSourceAndDestination(self, source, destination):
        source -= 1
        destination -= 1
        visited = ["false"] * self.n
        path = [source + 1]
        visited[source] = "true"
        cost = 0
        for i in range(1, self.n):
            nextNeighbour = self.nearest(source, visited)
            if nextNeighbour != source:
                cost += self.costsMatrix[source][nextNeighbour]
                source = nextNeighbour
                visited[nextNeighbour] = "true"
                path.append(nextNeighbour + 1)
            if source == destination:
                break
        return [len(path), path, cost]
