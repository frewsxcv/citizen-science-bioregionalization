"""Comparison against a published ecoregion framework.

The point of the project is to find ecological boundaries without being told
where they are. That only means something if the result can be checked against
a framework somebody else drew, so this scores a partition against EPA/CEC
Level II ecoregions.

Level II is the grain that can answer the question. Level I puts the whole
eastern United States in one region, so it cannot agree or disagree with a
north/south split; Level III is finer than anything this pipeline resolves.

The reference is checked in at `src/data/`, converted from the EPA shapefile by
`scripts/build_epa_reference.py`, so runs stay offline -- the same arrangement
as the Natural Earth coastline in `src/geocode.py`.
"""

import functools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Optional

import polars as pl
import polars_h3
import shapely
import shapely.geometry
from sklearn.metrics import adjusted_rand_score, v_measure_score

if TYPE_CHECKING:
    import shapely.strtree

logger = logging.getLogger(__name__)

EPA_GEOJSON_PATH = Path(__file__).parent / "data" / "epa_l2_east_coast.geojson"

#: The framework's code for open water. A hexagon whose centre lands here has no
#: terrestrial ecoregion, so it is dropped from the comparison rather than
#: treated as an eleventh region -- otherwise a run that kept coastal cells
#: would be scored partly on its ability to rediscover the sea.
WATER_CODE = "0.0"


class ReferenceAgreement(NamedTuple):
    """How well a partition agrees with the reference framework."""

    #: Adjusted Rand Index. Chance agreement is 0, identical partitions 1.
    #: Symmetric, and it does not care that the two partitions have different
    #: numbers of groups.
    adjusted_rand: float
    #: V-measure. Unlike ARI this rewards a partition that subdivides the
    #: reference consistently, which is what a finer cut of a nested hierarchy
    #: does, so the two move apart as k grows and the pair is more informative
    #: than either alone.
    v_measure: float
    #: Hexagons that carried both a cluster and a terrestrial reference region.
    compared: int
    #: Hexagons dropped because their centre was at sea or outside the
    #: reference extent.
    unmatched: int


@functools.lru_cache(maxsize=1)
def _reference_index() -> "tuple[shapely.strtree.STRtree, list, list[str]]":
    """Load the reference once and index it for point lookups."""
    with open(EPA_GEOJSON_PATH) as f:
        reference = json.load(f)
    polygons = [shapely.geometry.shape(f["geometry"]) for f in reference["features"]]
    codes = [f["properties"]["code"] for f in reference["features"]]
    logger.info(
        f"_reference_index: {len(polygons)} polygons across "
        f"{len(set(codes))} Level II regions"
    )
    return shapely.STRtree(polygons), polygons, codes


def reference_region_lf(geocode_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Assign each hexagon the reference region its centre falls in.

    The centre, not the hexagon's records. At H3 resolution 4 a cell centre can
    sit ~18 km from the records that produced it -- the mistake `src/geocode.py`
    documents at length for the coastline -- but a Level II region spans
    hundreds of kilometres, so the error is small against the thing being
    measured, and the cluster assignment this is compared with is itself per
    hexagon.

    Returns:
        `geocode` and `reference`, with `reference` null where the centre fell
        at sea or outside the reference extent.
    """
    centres = (
        geocode_lf.select("geocode")
        .with_columns(
            lat=polars_h3.cell_to_lat("geocode"),
            lng=polars_h3.cell_to_lng("geocode"),
        )
        .collect(engine="streaming")
    )

    tree, polygons, codes = _reference_index()
    assigned: list[Optional[str]] = []
    for lat, lng in zip(centres["lat"].to_list(), centres["lng"].to_list()):
        point = shapely.geometry.Point(lng, lat)
        code = None
        for candidate in tree.query(point):
            if polygons[candidate].contains(point):
                code = codes[candidate]
                break
        assigned.append(None if code == WATER_CODE else code)

    return centres.select("geocode").with_columns(
        reference=pl.Series("reference", assigned, dtype=pl.String)
    ).lazy()


def score_against_reference(
    geocode_cluster_df: pl.DataFrame,
    reference_lf: pl.LazyFrame,
) -> Optional[ReferenceAgreement]:
    """Score one partition against the reference framework.

    Args:
        geocode_cluster_df: `geocode` and `cluster`, one row per hexagon.
        reference_lf: The output of `reference_region_lf`.

    Returns:
        The agreement, or `None` when fewer than two hexagons carry a
        terrestrial reference region -- with one group there is nothing for
        either measure to be defined on.
    """
    joined = (
        geocode_cluster_df.lazy()
        .select("geocode", "cluster")
        .join(reference_lf, on="geocode", how="left")
        .collect(engine="streaming")
    )
    matched = joined.drop_nulls("reference")
    unmatched = joined.height - matched.height

    if matched.height < 2 or matched["reference"].n_unique() < 2:
        logger.info(
            f"score_against_reference: only {matched.height} hexagons matched a "
            f"terrestrial reference region; not scoring"
        )
        return None

    # Sorted before scoring. Both measures are invariant to label permutation
    # and to row order, so this is not for correctness -- it is so that the
    # numbers written into the findings page come out of a frame whose order is
    # fixed, which is the rule the rest of the pipeline follows.
    matched = matched.sort("geocode")
    labels = matched["cluster"].to_list()
    reference = matched["reference"].to_list()

    return ReferenceAgreement(
        adjusted_rand=float(adjusted_rand_score(reference, labels)),
        v_measure=float(v_measure_score(reference, labels)),
        compared=matched.height,
        unmatched=unmatched,
    )
