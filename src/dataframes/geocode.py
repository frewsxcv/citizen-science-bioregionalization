import polars as pl
import geohashr
from typing import Self
from src.data_container import DataContainer
from src.geocode import build_geohash_series_lazy
from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame


class GeocodeDataFrame(DataContainer):
    """
    A dataframe of unique, in-order geocodees that are connected to another known geocode.
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
        assert df.schema == self.SCHEMA
        self.df = df

    @classmethod
    def build(
        cls,
        darwin_core_csv_lazy_frame: DarwinCoreCsvLazyFrame,
        geocode_precision: int,
    ) -> Self:
        df = (
            darwin_core_csv_lazy_frame.lf.select("decimalLatitude", "decimalLongitude")
            .pipe(
                build_geohash_series_lazy,
                lat_col=pl.col("decimalLatitude"),
                lon_col=pl.col("decimalLongitude"),
                precision=geocode_precision,
            )
            .select("geocode")
            .unique()
            .sort(by="geocode")
            .collect()
        )

        df = df.with_columns(
            build_geocode_center_series(known_geocodees=df["geocode"]).alias(
                "center"
            ),
        )

        df = df.with_columns(
            build_geocode_neighbors_series(known_geocodees=df["geocode"]).alias(
                "neighbors"
            ),
        )

        return cls(df)


def build_geocode_center_series(known_geocodees: pl.Series) -> pl.Series:
    return pl.Series(
        [
            {
                "lat": lat,
                "lon": lon,
            }
            for lat, lon in (
                geohashr.decode(known_geocode) for known_geocode in known_geocodees
            )
        ],
        dtype=pl.Struct({"lat": pl.Float64(), "lon": pl.Float64()}),
    )


def build_geocode_neighbors_series(known_geocodees: pl.Series) -> pl.Series:
    return pl.Series(
        [
            [
                geocode
                for geocode in geohashr.neighbors(known_geocode).values()
                if geocode in known_geocodees
            ]
            for known_geocode in known_geocodees
        ],
        dtype=pl.List(pl.String()),
    )
