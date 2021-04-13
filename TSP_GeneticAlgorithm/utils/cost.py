def fitness_function(perm, param):
    n = param['noOfNodes']
    matrix = param['costs_matrix']

    overall_cost = 0
    for i in range(0, n-1):
        overall_cost += matrix[perm[i]][perm[i+1]]
    return overall_cost+matrix[perm[n-1]][perm[0]]
