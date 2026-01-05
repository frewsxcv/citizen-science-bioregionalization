import dataframely as dy
import polars as pl

from src.dataframes.taxonomy import TaxonomySchema


def mock_taxonomy_lf() -> dy.LazyFrame[TaxonomySchema]:
    """
    Creates a mock TaxonomyLazyFrame for testing.
    """
    taxonomy_data = [
        {
            "taxonId": 0,
            "phylum": "Chordata",
            "class": "Mammalia",
            "order": "Carnivora",
            "family": "Felidae",
            "genus": "Panthera",
            "species": "leo",
            "scientificName": "Panthera leo",
            "gbifTaxonId": 5219404,
        },
        {
            "taxonId": 1,
            "phylum": "Chordata",
            "class_": "Mammalia",
            "order": "Carnivora",
            "family": "Canidae",
            "genus": "Canis",
            "species": "lupus",
            "scientificName": "Canis lupus",
            "gbifTaxonId": 5219243,
        },
        {
            "taxonId": 2,
            "phylum": "Tracheophyta",
            "class": "Magnoliopsida",
            "order": "Fagales",
            "family": "Fagaceae",
            "genus": "Quercus",
            "species": "robur",
            "scientificName": "Quercus robur",
            "gbifTaxonId": 2878688,
        },
        {
            "taxonId": 3,
            "phylum": "Chordata",
            "class": "Aves",
            "order": "Anseriformes",
            "family": "",
            "genus": "",
            "species": "",
            "scientificName": "Anseriformes",
            "gbifTaxonId": 711,
        },
    ]
    taxonomy_df = pl.DataFrame(taxonomy_data).with_columns(
        pl.col("taxonId").cast(pl.UInt32),
        pl.col("phylum").cast(pl.Categorical),
        pl.col("class").cast(pl.Categorical),
        pl.col("order").cast(pl.Categorical),
        pl.col("family").cast(pl.Categorical),
        pl.col("genus").cast(pl.Categorical),
        pl.col("gbifTaxonId").cast(pl.UInt32),
    )
    return TaxonomySchema.validate(taxonomy_df).lazy()
