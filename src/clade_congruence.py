"""Do two unrelated biotas draw the same map?

Birds and plants share a landscape and almost nothing else -- different
dispersal, different generation times, different reasons to be where they are.
If clustering each one separately puts the boundary in the same place, the
boundary is a property of the landscape rather than of whichever taxa happen to
dominate the records.

This used to need one pipeline run per clade, with the numbers transcribed by
hand. Clustering is a function of the counts matrix the run has already built,
so each clade is re-clustered here instead, inside the run that reports it.

The cost is real but bounded: the expensive stages -- loading, geocoding,
counting -- are already done, and what repeats is the distance matrix and the
agglomerative fit over the same hexagons with fewer columns.
"""

import logging
from typing import NamedTuple, Optional

import polars as pl
from sklearn.metrics import adjusted_rand_score

from src.dataframes.geocode_cluster import build_geocode_cluster_multi_k_df
from src.dataframes.geocode_neighbors import build_geocode_neighbors_df
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.matrices.geocode_distance import CompositionMetric, GeocodeDistanceMatrix
from src.types import Reduction

logger = logging.getLogger(__name__)

#: Below this a clade cannot be clustered into a map worth comparing. Ward under
#: a contiguity constraint will happily return something for a handful of
#: hexagons; it just will not mean anything.
MIN_GEOCODES = 20


class CladePartition(NamedTuple):
    """One clade's own regionalization."""

    name: str
    #: `geocode`, `cluster`, `num_clusters` -- every k that was fit.
    multi_k_df: pl.DataFrame
    #: Hexagons that retained at least one of this clade's taxa.
    geocodes: int
    #: Taxa in the clade, after the run's own taxa filters.
    taxa: int


class Congruence(NamedTuple):
    """Agreement between two clades at one cut."""

    num_clusters: int
    adjusted_rand: float
    #: Hexagons both clades kept. Each clade drops hexagons where it has no
    #: records, so this is an intersection and is smaller than either partition.
    compared: int


def clade_taxon_ids(
    taxon_clade_lf: pl.LazyFrame,
    rank: str,
    name: str,
) -> pl.LazyFrame:
    """The taxonIds belonging to one clade."""
    return taxon_clade_lf.filter(pl.col(rank) == name).select("taxonId")


def cluster_clade(
    name: str,
    geocode_taxa_counts_lf: pl.LazyFrame,
    taxon_ids_lf: pl.LazyFrame,
    geocode_lf: pl.LazyFrame,
    min_k: int,
    max_k: int,
    seed: Optional[int],
    metric: CompositionMetric,
    reduction: Reduction,
) -> Optional[CladePartition]:
    """Cluster one clade on its own, over the hexagons where it occurs.

    The geocode set is narrowed rather than inherited whole. Restricting to a
    clade empties some hexagons entirely, and `GeocodeDistanceMatrix.build`
    asserts that its two inputs agree on the geocode set -- the same semi-join
    the notebook does after the taxa filters, for the same reason.

    Args:
        geocode_lf: The run's geocode frame, which carries the `center` column
            adjacency needs. Narrowed here rather than rebuilt, so the cell
            geometry is the one the rest of the run used.

    Returns:
        The clade's partition, or `None` when too few hexagons hold it.
    """
    counts_lf = geocode_taxa_counts_lf.join(taxon_ids_lf, on="taxonId", how="semi")

    occupied_lf = counts_lf.select("geocode").unique()
    clade_geocode_lf = (
        geocode_lf.join(occupied_lf, on="geocode", how="semi")
        # Sorted because the distance matrix indexes rows positionally and the
        # partitions compared downstream are matched by geocode, not by row.
        .sort("geocode")
    )
    geocode_df = clade_geocode_lf.collect(engine="streaming")
    taxa = counts_lf.select("taxonId").unique().select(pl.len()).collect().item()

    if geocode_df.height < MIN_GEOCODES:
        logger.info(
            f"cluster_clade: {name} occupies {geocode_df.height} hexagons, "
            f"below the {MIN_GEOCODES} needed to cluster it; skipping"
        )
        return None

    logger.info(
        f"cluster_clade: {name} over {geocode_df.height} hexagons and {taxa} taxa"
    )

    # Adjacency is rebuilt for this clade's hexagons. The run's own neighbours
    # frame describes a different, larger set, and a connectivity matrix has to
    # be square in the geocodes actually being clustered.
    neighbors_df = build_geocode_neighbors_df(geocode_df)
    connectivity = GeocodeConnectivityMatrix.build(neighbors_df)
    distance = GeocodeDistanceMatrix.build(
        counts_lf,
        clade_geocode_lf,
        random_state=seed,
        metric=metric,
        reduction=reduction,
    )

    # A clade cannot be cut into more regions than it has hexagons.
    clade_max_k = min(max_k, geocode_df.height - 1)
    if clade_max_k < min_k:
        logger.info(f"cluster_clade: {name} cannot support k={min_k}; skipping")
        return None

    multi_k_df = build_geocode_cluster_multi_k_df(
        clade_geocode_lf,
        distance,
        connectivity,
        min_k=min_k,
        max_k=clade_max_k,
    )
    return CladePartition(
        name=name, multi_k_df=multi_k_df, geocodes=geocode_df.height, taxa=taxa
    )


def congruence_by_k(a: CladePartition, b: CladePartition) -> list[Congruence]:
    """Agreement between two clades at every cut they share.

    Compared over the hexagons both clades occupy. Adjusted Rand is symmetric
    and corrects for chance, so a value near zero means the two maps agree no
    better than a coin would -- which is the null worth ruling out here.
    """
    shared_k = sorted(
        set(a.multi_k_df["num_clusters"].unique())
        & set(b.multi_k_df["num_clusters"].unique())
    )
    results = []
    for k in shared_k:
        left = a.multi_k_df.filter(pl.col("num_clusters") == k).select(
            "geocode", cluster_a="cluster"
        )
        right = b.multi_k_df.filter(pl.col("num_clusters") == k).select(
            "geocode", cluster_b="cluster"
        )
        joined = left.join(right, on="geocode", how="inner").sort("geocode")
        if joined.height < 2:
            continue
        results.append(
            Congruence(
                num_clusters=int(k),
                adjusted_rand=float(
                    adjusted_rand_score(
                        joined["cluster_a"].to_list(), joined["cluster_b"].to_list()
                    )
                ),
                compared=joined.height,
            )
        )
    return results


def latitude_spans(
    partition: CladePartition,
    num_clusters: int,
    geocode_centres: pl.DataFrame,
) -> pl.DataFrame:
    """Each cluster's latitude range at one cut.

    A north/south split shows up here as clusters with disjoint latitude
    ranges, which is the shape the East Coast result takes; a partition driven
    by sampling rather than biogeography gives clusters that all span the whole
    extent.
    """
    at_k = partition.multi_k_df.filter(pl.col("num_clusters") == num_clusters)
    return (
        at_k.join(geocode_centres, on="geocode", how="inner")
        .group_by("cluster")
        .agg(
            min_lat=pl.col("lat").min(),
            max_lat=pl.col("lat").max(),
            median_lat=pl.col("lat").median(),
            hexagons=pl.len(),
        )
        .sort("median_lat")
    )
