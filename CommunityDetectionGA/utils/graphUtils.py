import networkx as nx
import matplotlib.pyplot as plt
import warnings


def drawNetwork(graph, communities):
    warnings.simplefilter('ignore')
    pos = nx.spring_layout(graph)  # compute graph layout
    plt.figure(figsize=(15, 15))  # image is 8 x 8 inches
    nx.draw_networkx_nodes(graph, pos, node_size=600, node_color=communities)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color='black')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')

    plt.show()
