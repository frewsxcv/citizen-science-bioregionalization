import polars as pl
from src.dataframes.taxonomy import TaxonomyDataFrame


def mock_taxonomy_dataframe() -> TaxonomyDataFrame:
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
            "class": "Mammalia",
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
    taxonomy_df = pl.DataFrame(taxonomy_data, schema=TaxonomyDataFrame.SCHEMA)
    return TaxonomyDataFrame(taxonomy_df)