"""Emit the clustering at several levels of the tree rather than only at one.

A regionalization is conventionally reported as a nesting -- realms contain
regions contain provinces -- and Kreft & Jetz (2010) note the grain "will
usually be driven by the purpose of the study". This pipeline instead chose a
single k and discarded the rest of the tree, which is both less informative than
the literature's practice and, on this data, measurably worse.

Measured against EPA CEC Level II ecoregions over the published bounding box,
with the tree cut at each k:

    k=2   ARI 0.2552   V 0.3998
    k=3   ARI 0.3107   V 0.4583
    k=4   ARI 0.3108   V 0.4777
    k=8   ARI 0.2861   V 0.4773
    k=12  ARI 0.2416   V 0.4870
    k=15  ARI 0.2502   V 0.5098

ARI has an interior optimum and the two cuts at the top of it are 0.0001 apart,
which is a tie rather than a winner. V-measure rises with k throughout, because
it rewards subdividing the reference consistently, so it cannot choose either.
No single cut is right for every purpose -- which is the argument for publishing
the nesting instead of arguing about the cut.

The levels share a merge tree, so they are nested by construction: a cluster at
k=8 is contained in exactly one cluster at k=4.
"""

import json
import logging
from typing import Any, Sequence

import polars as pl

import bioregion_rs
from src.dataframes.cluster_boundary import build_cluster_boundary_df
from src.dataframes.cluster_color import build_cluster_color_df
from src.dataframes.cluster_neighbors import build_cluster_neighbors_df
from src.dataframes.cluster_significant_differences import (
    build_cluster_significant_differences_df,
)
from src.dataframes.cluster_taxa_statistics import build_cluster_taxa_statistics_df
from src.dataframes.geocode_cluster import build_geocode_cluster_df
from src.dataframes.significant_taxa_images import build_significant_taxa_images_df
from src.types import ClusterLevels

logger = logging.getLogger(__name__)


def default_ladder(min_k: int, max_k: int) -> list[int]:
    """The nesting to emit when the command line asks for no particular one.

    Doubling rather than consecutive. A regionalization is read as a nesting of
    realms inside regions inside provinces, and consecutive cuts of a merge tree
    differ by one split, which is not a change of grain a reader can see. Each
    level here roughly halves the mean region.

    Adaptive rather than a fixed list so that a run over a narrow range does not
    warn about levels it was never going to be able to emit.

    Args:
        min_k: Lowest level the tree was cut at.
        max_k: Highest level the tree was cut at.

    Returns:
        Sorted levels within [min_k, max_k], always non-empty.
    """
    levels = []
    k = max(min_k, 2)
    while k <= max_k:
        levels.append(k)
        k *= 2
    return levels or [min_k]


def resolve_levels(
    requested: Sequence[int] | None,
    optimal: int,
    min_k: int,
    max_k: int,
    display: int | None = None,
) -> list[int]:
    """Decide which cuts of the tree to emit.

    Always includes `optimal`, so the level the selector chose is present
    whatever else is asked for, and the single-level consumers still find it.

    Also includes `display` when given. The cut a consumer opens on has to be
    one of the cuts emitted, and with no `--hierarchy-levels` the only level
    that would otherwise be present is the selector's -- which is the one
    `defaults.DEFAULT_DISPLAY_LEVEL` exists to avoid opening on.

    Args:
        requested: Levels asked for on the command line, or None for just the
            selected one.
        optimal: The level the selector chose.
        min_k: Lowest level the tree was cut at.
        max_k: Highest level the tree was cut at.
        display: The level a consumer should open on, if it is in range.

    Returns:
        Sorted, de-duplicated levels, each within [min_k, max_k].
    """
    levels = set(requested or ())
    levels.add(optimal)
    if display is not None:
        levels.add(display)
    out_of_range = sorted(k for k in levels if not min_k <= k <= max_k)
    if out_of_range:
        logger.warning(
            f"Dropping hierarchy levels outside the range the tree was cut at "
            f"({min_k}-{max_k}): {out_of_range}"
        )
    return sorted(k for k in levels if min_k <= k <= max_k)


def resolve_default_level(
    preferred: int,
    optimal: int,
    levels: Sequence[int],
) -> int:
    """Decide which emitted cut a consumer should open on.

    Deliberately not the selector's `optimal`. See
    `defaults.DEFAULT_DISPLAY_LEVEL` for the evidence: the selector maximises a
    score silhouette dominates, silhouette falls with k here, and the cut it
    lands on scores worst against both references the run computes.

    Falls back to `optimal` when the preferred level was not emitted, so this
    can never point at a level that is not in the document.

    Args:
        preferred: The level to open on if it was emitted.
        optimal: The level the selector chose, used as the fallback.
        levels: The levels actually present.

    Returns:
        A level guaranteed to be in `levels`.
    """
    if preferred in levels:
        return preferred
    logger.info(
        f"resolve_default_level: level {preferred} was not emitted "
        f"(have {sorted(levels)}); opening on the selector's k={optimal}"
    )
    return optimal


def resolve_cluster_levels(
    requested: Sequence[int] | None,
    selector_k: int,
    min_k: int,
    max_k: int,
    preferred_display: int,
    pinned: bool = False,
) -> ClusterLevels:
    """Settle, once, which cuts a run emits and which one it publishes.

    The single place the k decision is made. It was previously split across two
    calls in the notebook whose results were then carried downstream as loose
    ints, which is how a cut got described under four different names -- see
    `types.ClusterLevels`.

    Args:
        requested: Levels asked for on the command line, or None to use
            `default_ladder`.
        selector_k: Where the combined score peaked, or the pinned k.
        min_k: Lowest level the tree was cut at.
        max_k: Highest level the tree was cut at.
        preferred_display: The level to publish if it was emitted; falls back
            to `selector_k`.
        pinned: Whether `selector_k` was asked for rather than found.

    Returns:
        A `ClusterLevels` whose `published` is guaranteed to be in `emitted`.
    """
    emitted = resolve_levels(
        # Falsy rather than `is None`: `--hierarchy-levels=` parses to an empty
        # list, which means "nothing asked for", not "emit nothing".
        requested or default_ladder(min_k, max_k),
        selector_k,
        min_k,
        max_k,
        display=preferred_display,
    )
    published = resolve_default_level(preferred_display, selector_k, emitted)
    levels = ClusterLevels(
        published=published,
        selector=selector_k,
        emitted=tuple(emitted),
        min_k=min_k,
        max_k=max_k,
        pinned=pinned,
    )
    logger.info(
        f"resolve_cluster_levels: publishing k={levels.published}, "
        f"selector {'pinned at' if pinned else 'peaked at'} k={levels.selector}, "
        f"emitting {list(levels.emitted)}"
    )
    return levels


def build_hierarchy_json(
    all_clusters_df: pl.DataFrame,
    geocode_lf: pl.LazyFrame,
    geocode_neighbors_df: pl.DataFrame,
    geocode_taxa_counts_lf: pl.LazyFrame,
    taxonomy_lf: pl.LazyFrame,
    levels: ClusterLevels,
    fetch_images: bool = True,
) -> str:
    """Build the frontend payload for every level in `levels`.

    Each level runs the same chain the single-level path runs -- neighbours,
    taxa statistics, significant differences, boundaries, colours, images -- and
    is serialised by the same Rust writer, so a level is byte-identical to what
    the single-level path would have produced for that k. They are then nested
    in one document rather than written separately.

    Args:
        all_clusters_df: The multi-k clustering, one row per (geocode, k).
        geocode_lf: Geocodes, in the order the distance matrix used.
        geocode_neighbors_df: Hexagon adjacency.
        geocode_taxa_counts_lf: Per-(hexagon, taxon) counts.
        taxonomy_lf: Taxonomy, providing names per taxonId.
        levels: The resolved cut decision: which k values to emit, and which
            one a consumer should open on.
        fetch_images: Passed through to the Wikidata lookup.

    Returns:
        A JSON document: `{"default_level": k, "levels": [{"k": k,
        "clusters": [...]}, ...]}`, where each `clusters` array is exactly the
        payload the single-level output produced.
    """
    taxonomy_df = taxonomy_lf.collect(engine="streaming")
    out: list[dict[str, Any]] = []

    for k in levels.emitted:
        logger.info(f"build_hierarchy_json: building level k={k}")
        geocode_cluster_df = build_geocode_cluster_df(all_clusters_df, k)
        cluster_neighbors_df = build_cluster_neighbors_df(
            geocode_neighbors_df, geocode_cluster_df
        )
        cluster_neighbors_lf = cluster_neighbors_df.lazy()
        cluster_taxa_statistics_df = build_cluster_taxa_statistics_df(
            geocode_taxa_counts_lf, geocode_cluster_df.lazy(), taxonomy_lf
        )
        cluster_significant_differences_df = build_cluster_significant_differences_df(
            cluster_taxa_statistics_df, cluster_neighbors_lf
        )
        cluster_boundary_df = build_cluster_boundary_df(geocode_cluster_df, geocode_lf)
        cluster_colors_df = build_cluster_color_df(
            cluster_neighbors_lf, cluster_taxa_statistics_df
        )
        significant_taxa_images_df = build_significant_taxa_images_df(
            cluster_significant_differences_df, taxonomy_df, fetch_images=fetch_images
        )
        clusters = json.loads(
            bioregion_rs.build_json_output(
                cluster_significant_differences_df,
                cluster_boundary_df,
                taxonomy_df,
                cluster_colors_df,
                significant_taxa_images_df,
            )
        )
        logger.info(
            f"build_hierarchy_json: level k={k} has {len(clusters)} clusters"
        )
        out.append({"k": k, "clusters": clusters})

    return json.dumps({"default_level": levels.published, "levels": out})
