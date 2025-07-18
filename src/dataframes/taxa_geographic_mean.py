import polars as pl
from polars_darwin_core.darwin_core import KingdomDataType  # type: ignore
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsDataFrame
import logging
from src.data_container import DataContainer
import polars_h3  # type: ignore

logger = logging.getLogger(__name__)


class TaxaGeographicMeanDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "kingdom": KingdomDataType,
        "taxonRank": pl.String(),
        "scientificName": pl.String(),
        "mean_lat": pl.Float64(),
        "mean_lon": pl.Float64(),
    }

    def __init__(self, df: pl.DataFrame):
        self.df = df

    @classmethod
    def build(
        cls, geocode_taxa_counts: GeocodeTaxaCountsDataFrame
    ) -> "TaxaGeographicMeanDataFrame":
        # TODO: this doesn't handle the international date line
        df = (
            geocode_taxa_counts.df.lazy()
            .with_columns(
                (pl.col("lat") * pl.col("count")).alias("lat_scaled"),
                (pl.col("lon") * pl.col("count")).alias("lon_scaled"),
            )
            .group_by("kingdom", "taxonRank", "scientificName")
            .agg(
                (pl.col("lat_scaled").sum() / pl.col("count").sum()).alias("mean_lat"),
                (pl.col("lon_scaled").sum() / pl.col("count").sum()).alias("mean_lon"),
            )
        )
        return cls(df.collect())
