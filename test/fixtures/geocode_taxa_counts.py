import polars as pl
import dataframely as dy
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsSchema


def mock_geocode_taxa_counts_dataframe() -> dy.DataFrame[GeocodeTaxaCountsSchema]:
    """
    Creates a mock GeocodeTaxaCountsDataFrame for testing.
    """
    geocode_taxa_counts_data = [
        {"geocode": 1000, "taxonId": 0, "count": 5},  # Panthera leo
        {"geocode": 1000, "taxonId": 1, "count": 3},  # Canis lupus
        {"geocode": 2000, "taxonId": 0, "count": 2},  # Panthera leo
        {"geocode": 2000, "taxonId": 2, "count": 8},  # Quercus robur
        {"geocode": 3000, "taxonId": 2, "count": 4},  # Quercus robur
        {"geocode": 3000, "taxonId": 3, "count": 6},  # Anseriformes
    ]
    geocode_taxa_counts_df = pl.DataFrame(geocode_taxa_counts_data).with_columns(
        pl.col("geocode").cast(pl.UInt64),
        pl.col("taxonId").cast(pl.UInt32),
        pl.col("count").cast(pl.UInt32),
    )
    return GeocodeTaxaCountsSchema.validate(geocode_taxa_counts_df)