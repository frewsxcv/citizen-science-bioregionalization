import polars as pl
import dataframely as dy
from polars_darwin_core import Kingdom
from src.dataframes.taxonomy import TaxonomySchema


def mock_taxonomy_dataframe() -> dy.DataFrame[TaxonomySchema]:
    """
    Creates a mock TaxonomyDataFrame for testing.
    """
    taxonomy_data = [
        {
            "taxonId": 0,
            "kingdom": "Animalia",
            "phylum": "Chordata",
            "class": "Mammalia",
            "order": "Carnivora",
            "family": "Felidae",
            "genus": "Panthera",
            "species": "leo",
            "taxonRank": "species",
            "scientificName": "Panthera leo",
        },
        {
            "taxonId": 1,
            "kingdom": "Animalia",
            "phylum": "Chordata",
            "class_": "Mammalia",
            "order": "Carnivora",
            "family": "Canidae",
            "genus": "Canis",
            "species": "lupus",
            "taxonRank": "species",
            "scientificName": "Canis lupus",
        },
        {
            "taxonId": 2,
            "kingdom": "Plantae",
            "phylum": "Tracheophyta",
            "class": "Magnoliopsida",
            "order": "Fagales",
            "family": "Fagaceae",
            "genus": "Quercus",
            "species": "robur",
            "taxonRank": "species",
            "scientificName": "Quercus robur",
        },
        {
            "taxonId": 3,
            "kingdom": "Animalia",
            "phylum": "Chordata",
            "class": "Aves",
            "order": "Anseriformes",
            "family": "",
            "genus": "",
            "species": "",
            "taxonRank": "order",
            "scientificName": "Anseriformes",
        },
    ]
    taxonomy_df = pl.DataFrame(taxonomy_data).with_columns(
        pl.col("taxonId").cast(pl.UInt32),
        pl.col("kingdom").cast(pl.Enum(Kingdom)),
        pl.col("phylum").cast(pl.Categorical),
        pl.col("class").cast(pl.Categorical),
        pl.col("order").cast(pl.Categorical),
        pl.col("family").cast(pl.Categorical),
        pl.col("genus").cast(pl.Categorical),
        pl.col("taxonRank").cast(pl.Categorical),
    )
    return TaxonomySchema.validate(taxonomy_df)