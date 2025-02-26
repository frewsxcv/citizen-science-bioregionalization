from inspect import signature
import mermaid as md

from src.data_container import DataContainer
from src.dataframes import (
    cluster_color,
    cluster_taxa_statistics,
    geohash_cluster,
    geohash_species_counts,
    taxonomy,
    taxa_geographic_mean,
    cluster_significant_differences,
)
from src.lazyframes import darwin_core_csv
from src.matrices import distance, connectivity
from src.series import geohash


def build_mermaid_graph() -> md.Mermaid:
    script = "flowchart TD\n"
    classes = set(DataContainer.__subclasses__())
    for class_ in classes:
        for argument in signature(class_.build).parameters.values():
            if argument.annotation not in classes:
                continue

            # Output in mermaid syntax
            script += f"  {argument.annotation.__name__} --> {class_.__name__}\n"
    return md.Mermaid(md.Graph(title="Signatures", script=script))
