import polars as pl
import geohashr
from typing import Self
from src.data_container import DataContainer
from src.geohash import build_geohash_series_lazy
from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame


class GeohashDataFrame(DataContainer):
    """
    A dataframe of unique, in-order geohashes that are connected to another known geohash.
    """

    # TODO: Only keep the largest connected component

    df: pl.DataFrame

    SCHEMA = {
        "geohash": pl.String(),
        "neighbors": pl.List(pl.String()),
    }

    def __init__(self, df: pl.DataFrame):
        assert df.schema == self.SCHEMA
        self.df = df

    @classmethod
    def build(
        cls,
        darwin_core_csv_lazy_frame: DarwinCoreCsvLazyFrame,
        geohash_precision: int,
    ) -> Self:
        df = (
            darwin_core_csv_lazy_frame.lf
            .select("decimalLatitude", "decimalLongitude")
            .pipe(
                build_geohash_series_lazy,
                lat_col=pl.col("decimalLatitude"),
                lon_col=pl.col("decimalLongitude"),
                precision=geohash_precision,
            )
            .select("geohash")
            .unique()
            .sort(by="geohash")
            .collect()
        )

        df = df.with_columns(
            build_geohash_neighbors_series(known_geohashes=df["geohash"]).alias(
                "neighbors"
            ),
        )

        # Filter out geohashes that don't have neighbors
        df = df.filter(pl.col("neighbors").list.len() > 0)

        return cls(df)


def build_geohash_neighbors_series(known_geohashes: pl.Series) -> pl.Series:
    return pl.Series(
        [
            [
                geohash
                for geohash in geohashr.neighbors(known_geohash).values()
                if geohash in known_geohashes
            ]
            for known_geohash in known_geohashes
        ],
        dtype=pl.List(pl.String()),
    )
