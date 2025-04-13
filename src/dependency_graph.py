from inspect import signature
import networkx as nx

from src.data_container import DataContainer
from src import *
from src.dataframes import *
from src.lazyframes import *
from src.matrices import *
from src.output import *



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


def plot_dependency_graph() -> str:
    """
    Generates a mermaid representation of the dependency graph.

    Returns:
        str: Mermaid formatted graph string
    """
    G = build_dependency_graph()

    # Create mermaid graph string
    mermaid_str = "graph TD;\n"

    # Add edges (which implicitly define nodes)
    for source, target in G.edges():
        mermaid_str += f"    {source}-->{target};\n"

    # If there are isolated nodes with no edges, add them explicitly
    for node in G.nodes():
        if G.degree(node) == 0:
            mermaid_str += f"    {node};\n"

    return mermaid_str
