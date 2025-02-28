from enum import Enum
import polars as pl


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
