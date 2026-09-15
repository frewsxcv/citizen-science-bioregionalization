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
    sample_per_hexagon: int = 2000,
    min_land_share: float = 0.5,
) -> pl.LazyFrame:
    """Drop hexagons whose records are mostly at sea.

    Country-code and bounding-box filtering both include a country's maritime
    zone, so coastal clusters can otherwise be driven by fish and seabirds
    rather than terrestrial biota.

    The test is the share of a hexagon's own records that fall on land. Two
    earlier versions instead tested a single synthetic point, and both were
    wrong for the same reason -- a point derived from the data need not sit
    where the data is:

    - the hexagon's geometric centre. At H3 resolution 4 a cell spans roughly
      1,770 km2, so its midpoint can be 18 km from the records. Manhattan's
      cell centres in Long Island Sound, 17.7 km from Central Park.
    - the marginal median of the records. Median latitude and median longitude
      are computed independently, so the resulting point need not be near any
      actual record. On Manhattan -- a narrow island between two rivers -- it
      lands in the East River, while 88.0% of the cell's ten million records
      are on land.

    The share has no such failure mode: 88% on land keeps the cell whatever
    shape it is. It also states the question directly rather than proxying it.

    Records are sampled rather than all tested, because the published run
    carries hundreds of millions of them and the answer does not need that
    precision. The sample is shuffled with a fixed seed rather than taken from
    the head, which matters more than it sounds: the snapshot is ordered by
    source dataset, so a cell's first records all come from whichever dataset
    happens to appear earliest. On Manhattan the first 2,000 records are 45.8%
    on land against a true 88.9%, which is the difference between dropping the
    cell and keeping it. Seeded shuffling is both deterministic and
    representative; ordering alone is only the former.

    Args:
        lf: Occurrence records with decimalLatitude/decimalLongitude.
        geocode_precision: H3 resolution; must match the run's precision.
        sample_per_hexagon: Records per hexagon to test against the coastline.
        min_land_share: Share that must be on land for the hexagon to be kept.

    Returns:
        The input restricted to hexagons whose records are mostly on land.
    """
    geocode_expr = polars_h3.latlng_to_cell(
        "decimalLatitude",
        "decimalLongitude",
        resolution=geocode_precision,
        return_dtype=pl.UInt64,
    )
    with_geocode = lf.with_columns(geocode_expr.alias("_geocode"))

    sampled = (
        with_geocode.select("_geocode", "decimalLatitude", "decimalLongitude")
        .group_by("_geocode")
        .agg(
            # As a struct so the coordinate pair survives the shuffle together;
            # shuffling the two columns separately would pair each latitude with
            # somebody else's longitude.
            pl.struct(["decimalLatitude", "decimalLongitude"])
            .shuffle(seed=0)
            .head(sample_per_hexagon)
            .alias("_sample")
        )
        .explode("_sample")
        .unnest("_sample")
        .collect(engine="streaming")
    )
    if sampled.height == 0:
        return with_geocode.drop("_geocode")

    tree, _polygons = _land_index()
    points = shapely.points(
        sampled["decimalLongitude"].to_numpy(), sampled["decimalLatitude"].to_numpy()
    )
    # `intersects` is symmetric, so it does not depend on which side of the
    # predicate the tree geometry lands on, and it counts a point exactly on the
    # coastline as land.
    hits = tree.query(points, predicate="intersects")
    on_land = np.zeros(sampled.height, dtype=bool)
    on_land[np.unique(hits[0])] = True

    terrestrial = (
        sampled.with_columns(_on_land=pl.Series(on_land))
        .group_by("_geocode")
        .agg(pl.col("_on_land").mean().alias("_land_share"))
        .filter(pl.col("_land_share") >= min_land_share)
        .select("_geocode")
    )
    return with_geocode.join(terrestrial.lazy(), on="_geocode", how="semi").drop(
        "_geocode"
    )


def adaptive_min_hex_records(
    lf: pl.LazyFrame,
    geocode_precision: int,
    absolute_floor: int,
    median_fraction: float,
    ceiling: int,
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

    The ceiling matters at the other end, and was found the hard way: the
    published East Coast run is at H3 resolution 4, where a hexagon covers seven
    times the area of a resolution-5 one, and its median holds around 73,000
    records. A tenth of that is 7,274 -- a "sparse hexagon" filter discarding
    hexagons with thousands of observations, which is not what this is for.
    Every region measured while choosing the rule derived a floor between 20 and
    34, so the ceiling bounds the failure without touching any of them.

    Args:
        lf: Occurrence records with decimalLatitude/decimalLongitude.
        geocode_precision: H3 resolution; must match the run's precision.
        absolute_floor: Never return less than this.
        median_fraction: Share of the median hexagon's record count to require.
        ceiling: Never return more than this, however dense the region.

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
    scaled = int(round(median_fraction * float(median)))  # type: ignore[arg-type]
    return min(ceiling, max(absolute_floor, scaled))
