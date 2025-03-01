import polars as pl
from src.darwin_core import kingdom_enum
from src.dataframes.geohash_taxa_counts import GeohashTaxaCountsDataFrame
from src.geohash import geohash_to_lat_lon, geohash_to_lat_lon_lazy
from typing import Self
import logging
from src.data_container import DataContainer

logger = logging.getLogger(__name__)


class TaxaGeographicMeanDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "kingdom": kingdom_enum,
        "taxonRank": pl.String(),
        "scientificName": pl.String(),
        "mean_lat": pl.Float64(),
        "mean_lon": pl.Float64(),
    }

    def __init__(self, df: pl.DataFrame):
        self.df = df

    @classmethod
    def build(cls, geohash_taxa_counts: GeohashTaxaCountsDataFrame) -> Self:
        # TODO: this doesn't handle the international date line
        df = (
            geohash_taxa_counts.df.lazy()
            .pipe(geohash_to_lat_lon_lazy, pl.col("geohash"))
            .with_columns(
                (pl.col("geohash_lat") * pl.col("count")).alias("lat_scaled"),
                (pl.col("geohash_lon") * pl.col("count")).alias("lon_scaled"),
            )
            .group_by("kingdom", "taxonRank", "scientificName")
            .agg(
                (pl.col("lat_scaled").sum() / pl.col("count").sum()).alias("mean_lat"),
                (pl.col("lon_scaled").sum() / pl.col("count").sum()).alias("mean_lon"),
            )
        )
        return cls(df.collect())
