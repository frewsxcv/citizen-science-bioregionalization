import geohashr
import polars as pl
from src.bbox import Bbox
from shapely.geometry import Point
from src.types import Geohash


def geohash_to_bbox(geohash: str) -> Bbox:
    lat, lon, lat_err, lon_err = geohashr.decode_exact(geohash)
    return Bbox(
        sw=Point(lon - lon_err, lat - lat_err),
        ne=Point(lon + lon_err, lat + lat_err),
    )


def build_geohash_series(
    data_frame: pl.DataFrame, lat_col: pl.Expr, lon_col: pl.Expr, precision: int
) -> pl.DataFrame:
    return data_frame.with_columns(
        pl.struct([lat_col.alias("lat"), lon_col.alias("lon")])
        .map_elements(
            lambda series: geohashr.encode(series["lat"], series["lon"], precision),
            return_dtype=pl.String,
        )
        .alias("geohash")
    )


def build_geohash_series_lazy(
    lazy_frame: pl.LazyFrame, lat_col: pl.Expr, lon_col: pl.Expr, precision: int
) -> pl.LazyFrame:
    return lazy_frame.with_columns(
        pl.struct([lat_col.alias("lat"), lon_col.alias("lon")])
        .map_elements(
            lambda series: geohashr.encode(series["lat"], series["lon"], precision),
            return_dtype=pl.String,
        )
        .alias("geohash")
    )


def geohash_to_lat_lon(df: pl.DataFrame, geohash_col: pl.Expr) -> pl.DataFrame:
    return df.with_columns(
        geohash_col.map_elements(
            lambda geohash: list(geohashr.decode(geohash)),
            return_dtype=pl.List(pl.Float64),
        ).alias("lat_lon").list.to_array(2)
    )


def geohash_to_lat_lon_lazy(lazy_frame: pl.LazyFrame, geohash_col: pl.Expr) -> pl.LazyFrame:
    return lazy_frame.with_columns(
        geohash_col.map_elements(
            lambda geohash: list(geohashr.decode(geohash)),
            return_dtype=pl.List(pl.Float64),
        ).alias("lat_lon").list.to_array(2)
    )


def is_water(geohash: Geohash) -> bool:
    return geohash in ["9ny", "9nz", "9vj", "f04", "dpv", "9ug", "dqg", "9pw"]
