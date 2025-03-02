import polars as pl
from typing import Self
from src.data_container import DataContainer
from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame
import polars_h3


class GeocodeDataFrame(DataContainer):
    """
    A dataframe of unique, in-order geocodes that are connected to another known geocode.
    """

    df: pl.DataFrame

    SCHEMA = {
        "geocode": pl.String(),
        "center": pl.Struct(
            {
                "lat": pl.Float64(),
                "lon": pl.Float64(),
            }
        ),
        "neighbors": pl.List(pl.String()),
    }

    def __init__(self, df: pl.DataFrame):
        assert (
            df.schema == self.SCHEMA
        ), f"Schema mismatch: {df.schema} != {self.SCHEMA}"
        self.df = df

    @classmethod
    def build(
        cls,
        darwin_core_csv_lazy_frame: DarwinCoreCsvLazyFrame,
        geocode_precision: int,
    ) -> Self:
        df = (
            darwin_core_csv_lazy_frame.lf.select("decimalLatitude", "decimalLongitude")
            .with_columns(
                polars_h3.latlng_to_cell(
                    "decimalLatitude",
                    "decimalLongitude",
                    resolution=geocode_precision,
                    return_dtype=pl.Utf8,
                ).alias("geocode"),
            )
            .select("geocode")
            .unique()
            .sort(by="geocode")
            .with_columns(
                polars_h3.cell_to_latlng(pl.col("geocode"))
                .list.to_struct(fields=["lat", "lon"])
                .alias("center")
            )
            .collect()
        )

        df = df.with_columns(
            allNeighbors=polars_h3.grid_ring(pl.col("geocode"), 1),
            knownGeocodes=pl.lit(
                df["geocode"].unique().to_list(), dtype=pl.List(pl.Utf8)
            ),
        ).with_columns(
            neighbors=pl.col("allNeighbors").list.set_intersection(
                pl.col("knownGeocodes")
            ),
        )

        return cls(df.select(cls.SCHEMA.keys()))
