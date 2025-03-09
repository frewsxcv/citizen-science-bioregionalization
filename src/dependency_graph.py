from inspect import signature
import networkx as nx
import matplotlib.pyplot as plt

from src.data_container import DataContainer
from run import *


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
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")

    nx.draw_networkx_nodes(G, pos, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    plt.title("DataContainer Dependency Graph")
    plt.axis("off")
    plt.show()
