"""Constants for Darwin Core data processing.

This module defines constants used throughout the citizen science
bioregionalization project, particularly for handling Darwin Core
occurrence data.
"""

import polars as pl

# Valid kingdom values in Darwin Core taxonomy
# Based on the GBIF backbone taxonomy
KINGDOM_VALUES: list[str] = [
    "Animalia",
    "Plantae",
    "Fungi",
    "Protozoa",
    "Chromista",
    "Archaea",
    "Bacteria",
    "Viruses",
    "incertae sedis",
]

# Polars Enum data type for kingdom column
KINGDOM_DATA_TYPE: pl.Enum = pl.Enum(KINGDOM_VALUES)
