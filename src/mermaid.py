from inspect import signature
import networkx as nx
import matplotlib.pyplot as plt

from src.data_container import DataContainer
from src.dataframes import (
    cluster_color,
    cluster_neighbors,
    cluster_taxa_statistics,
    geocode_boundary,
    geocode_cluster,
    geocode_taxa_counts,
    taxonomy,
    taxa_geographic_mean,
    cluster_significant_differences,
    geocode,
)
from src.lazyframes import darwin_core_csv
from src.matrices import distance, connectivity


def build_dependency_graph():
    G = nx.DiGraph()

    classes = set(DataContainer.__subclasses__())
    for class_ in classes:
        G.add_node(class_.__name__)
        for argument in signature(class_.build).parameters.values():
            if argument.annotation not in classes:
                continue
            G.add_edge(argument.annotation.__name__, class_.__name__)

    return G


def plot_dependency_graph():
    G = build_dependency_graph()

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    plt.title("DataContainer Dependency Graph")
    plt.axis('off')
    plt.show()
