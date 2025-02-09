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


class TaxonRank(Enum):
    """
    Each of these values is also the name of a Darwin Core column
    """

    phylum = "phylum"
    class_ = "class"
    order = "order"
    family = "family"
    genus = "genus"
    species = "species"
