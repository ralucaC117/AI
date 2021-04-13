class Ant:
    def __init__(self, startingNode=None, costsMatrix=None):
        self.__visited = []
        self.__visited.append(startingNode)
        self.__trailLength = 0
        self.__costsMatrix = costsMatrix

    @property
    def visited(self):
        return self.__visited

    def addNodeToVisited(self, node):
        self.__visited.append(node)

    @property
    def trailLength(self):
        length = 0
        for node in range(len(self.__visited) - 1):
            length += self.__costsMatrix[self.__visited[node]][self.__visited[node + 1]]
        length += self.__costsMatrix[self.__visited[-1]][self.__visited[0]]
        return length

