def modularity(communities, param):
    noNodes = param['noNodes']
    mat = param['mat']
    dict_of_nodes = param['dict_of_nodes']
    degrees = param['degrees']
    noEdges = param['noEdges']
    nodes = param['graph'].nodes
    M = 2 * noEdges
    Q = 0.0
    for node_i in dict_of_nodes:
        for j, node_j in enumerate(nodes):
            if communities[dict_of_nodes[node_i]] == communities[j]:
                Q += (mat[dict_of_nodes[node_i], j] - degrees[node_i] * degrees[node_j] / M)
    return Q * 1 / M
