import dataframely as dy
import polars as pl

from src.constants import KINGDOM_VALUES
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
            "gbifTaxonId": 5219404,
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
            "gbifTaxonId": 5219243,
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
            "gbifTaxonId": 2878688,
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
            "gbifTaxonId": 711,
        },
    ]
    taxonomy_df = pl.DataFrame(taxonomy_data).with_columns(
        pl.col("taxonId").cast(pl.UInt32),
        pl.col("kingdom").cast(pl.Enum(KINGDOM_VALUES)),
        pl.col("phylum").cast(pl.Categorical),
        pl.col("class").cast(pl.Categorical),
        pl.col("order").cast(pl.Categorical),
        pl.col("family").cast(pl.Categorical),
        pl.col("genus").cast(pl.Categorical),
        pl.col("taxonRank").cast(pl.Categorical),
        pl.col("gbifTaxonId").cast(pl.UInt32),
    )
    return TaxonomySchema.validate(taxonomy_df)
