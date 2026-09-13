import functools
import json
from pathlib import Path

import numpy as np
import polars as pl
import polars_h3
import shapely

from src.types import Bbox

LAND_GEOJSON_PATH = Path(__file__).parent / "data" / "ne_50m_land.geojson"


def with_geocode_lf(lf: pl.LazyFrame, geocode_precision: int) -> pl.LazyFrame:
    """Geocodes a lazy frame with decimalLatitude and decimalLongitude columns."""
    return lf.with_columns(
        polars_h3.latlng_to_cell(
            "decimalLatitude",
            "decimalLongitude",
            resolution=geocode_precision,
            return_dtype=pl.UInt64,
        ).alias("geocode"),
    )


def filter_by_bounding_box(
    lf: pl.LazyFrame,
    bounding_box: Bbox,
    lat_col: str = "decimalLatitude",
    lng_col: str = "decimalLongitude",
) -> pl.LazyFrame:
    """Filter a LazyFrame to rows within a geographic bounding box.

    Args:
        lf: The LazyFrame to filter.
        bounding_box: Geographic bounding box to filter records.
        lat_col: Name of the latitude column.
        lng_col: Name of the longitude column.

    Returns:
        A LazyFrame filtered to rows with valid coordinates within the bounding box.
    """
    return lf.filter(
        pl.col(lat_col).is_not_null()
        & pl.col(lng_col).is_not_null()
        & pl.col(lat_col).is_between(bounding_box.min_lat, bounding_box.max_lat)
        & pl.col(lng_col).is_between(bounding_box.min_lng, bounding_box.max_lng)
    )


def select_geocode_lf(lf: pl.LazyFrame, geocode_precision: int) -> pl.LazyFrame:
    """Geocodes a lazy frame with decimalLatitude and decimalLongitude columns."""
    return lf.select(
        geocode=polars_h3.latlng_to_cell(
            "decimalLatitude",
            "decimalLongitude",
            resolution=geocode_precision,
            return_dtype=pl.UInt64,
        ),
    )


def filter_sparse_geocodes_lf(
    lf: pl.LazyFrame,
    geocode_precision: int,
    min_records: int,
) -> pl.LazyFrame:
    """Drop occurrences in hexagons holding fewer than `min_records` records.

    A hexagon observed once yields a composition vector of a single taxon, which
    is maximally distant from everything else and says more about where people
    looked than about what lives there. Citizen-science effort is concentrated
    enough that these dominate the long tail: in a country-scale run at
    precision 5, a quarter of hexagons held fewer than 8 records against a
    median of 133.

    Applied to the occurrence records rather than to the geocodes, so that the
    geocode set and the taxa counts are both derived from the same rows and
    agree by construction.

    Args:
        lf: Occurrence records with decimalLatitude/decimalLongitude.
        geocode_precision: H3 resolution; must match the run's precision.
        min_records: Minimum records a hexagon must hold to be kept.

    Returns:
        The input restricted to sufficiently sampled hexagons.
    """
    with_geocode = lf.with_columns(
        polars_h3.latlng_to_cell(
            "decimalLatitude",
            "decimalLongitude",
            resolution=geocode_precision,
            return_dtype=pl.UInt64,
        ).alias("_geocode")
    )
    well_sampled = (
        with_geocode.group_by("_geocode")
        .len()
        .filter(pl.col("len") >= min_records)
        .select("_geocode")
    )
    return with_geocode.join(well_sampled, on="_geocode", how="semi").drop("_geocode")


@functools.lru_cache(maxsize=1)
def _land_index() -> "tuple[shapely.STRtree, list]":
    """Load the coastline once and index it for point lookups.

    Natural Earth 1:50m land polygons, checked in at src/data/ so that runs stay
    offline. At that scale the coastline is generalised to roughly 50 km, which
    is coarse relative to an H3 resolution-5 hexagon (~250 km²) -- expect
    disagreement on individual coastal cells, not on whether a region is
    offshore.
    """
    with open(LAND_GEOJSON_PATH) as f:
        land = json.load(f)
    polygons = [shapely.geometry.shape(f["geometry"]) for f in land["features"]]
    return shapely.STRtree(polygons), polygons


def filter_terrestrial_geocodes_lf(
    lf: pl.LazyFrame,
    geocode_precision: int,
) -> pl.LazyFrame:
    """Drop occurrences whose hexagon centroid falls in the sea.

    Country-code filtering includes a country's maritime zone, so coastal
    clusters can be driven by fish and seabirds rather than terrestrial biota.
    The unit of the test is the hexagon centroid rather than the occurrence,
    because the goal is to drop whole ocean hexagons, not marine records that
    happen to sit in an otherwise terrestrial cell.

    Args:
        lf: Occurrence records with decimalLatitude/decimalLongitude.
        geocode_precision: H3 resolution; must match the run's precision.

    Returns:
        The input restricted to hexagons centred on land.
    """
    geocode_expr = polars_h3.latlng_to_cell(
        "decimalLatitude",
        "decimalLongitude",
        resolution=geocode_precision,
        return_dtype=pl.UInt64,
    )
    with_geocode = lf.with_columns(geocode_expr.alias("_geocode"))

    centroids = (
        with_geocode.select("_geocode")
        .unique()
        .with_columns(
            lat=polars_h3.cell_to_lat("_geocode"),
            lng=polars_h3.cell_to_lng("_geocode"),
        )
        .collect(engine="streaming")
    )

    tree, polygons = _land_index()
    points = shapely.points(centroids["lng"].to_numpy(), centroids["lat"].to_numpy())
    # `intersects` is symmetric, so it does not depend on which side of the
    # predicate the tree geometry lands on, and it counts a point exactly on the
    # coastline as land.
    hits = tree.query(points, predicate="intersects")
    on_land = np.zeros(centroids.height, dtype=bool)
    on_land[hits[0]] = True

    terrestrial = centroids.filter(pl.Series(on_land)).select("_geocode")
    return (
        with_geocode.join(terrestrial.lazy(), on="_geocode", how="semi")
        .drop("_geocode")
    )


def adaptive_min_hex_records(
    lf: pl.LazyFrame,
    geocode_precision: int,
    absolute_floor: int,
    median_fraction: float,
) -> int:
    """Derive a sampling floor from how densely this region was surveyed.

    A fixed floor cannot serve every extent. Measured across four regions, a
    flat 50 records repaired Colombia, California and southeast Australia but
    made the Alps worse: the Alps are evenly surveyed, have no tail of
    under-sampled hexagons, and a floor there only discards data. Scaling with
    the region's own median handles both, and keeps more hexagons than the flat
    floor everywhere it was measured.

    The absolute term matters where the whole extent is thin. California's
    median hexagon held 25 records, so a purely relative floor came out at 2 and
    filtered nothing, leaving the partition at chance agreement under
    perturbation.

    Args:
        lf: Occurrence records with decimalLatitude/decimalLongitude.
        geocode_precision: H3 resolution; must match the run's precision.
        absolute_floor: Never return less than this.
        median_fraction: Share of the median hexagon's record count to require.

    Returns:
        The record count a hexagon must reach to be kept.
    """
    per_hexagon = (
        lf.select(
            polars_h3.latlng_to_cell(
                "decimalLatitude",
                "decimalLongitude",
                resolution=geocode_precision,
                return_dtype=pl.UInt64,
            ).alias("_geocode")
        )
        .group_by("_geocode")
        .len()
        .collect(engine="streaming")["len"]
    )
    median = per_hexagon.median()
    if median is None:
        return absolute_floor
    return max(
        absolute_floor,
        int(round(median_fraction * float(median))),  # type: ignore[arg-type]
    )
