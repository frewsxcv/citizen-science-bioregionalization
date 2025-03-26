from enum import Enum
import polars as pl
from typing import List


kingdom_enum = pl.Enum(
    [
        "Animalia",
        "Archaea",
        "Bacteria",
        "Chromista",
        "Fungi",
        "Plantae",
        "Protozoa",
        "Viruses",
        "incertae sedis",
    ],
)

# Taxonomic ranks in hierarchical order
TAXONOMIC_RANKS: List[str] = [
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]
