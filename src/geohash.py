import geohashr
import polars as pl
from src.bbox import Bbox
from src.point import Point
from src.types import Geohash


def geohash_to_bbox(geohash: str) -> Bbox:
    lat, lon, lat_err, lon_err = geohashr.decode_exact(geohash)
    return Bbox(
        sw=Point(lat=lat - lat_err, lon=lon - lon_err),
        ne=Point(lat=lat + lat_err, lon=lon + lon_err),
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


def is_water(geohash: Geohash) -> bool:
    return geohash in ["9ny", "9nz", "9vj", "f04", "dpv", "9ug", "dqg", "9pw"]
