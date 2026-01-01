import logging
from typing import Iterator, List, Tuple

import dataframely as dy
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering

from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.matrices.geocode_distance import GeocodeDistanceMatrix
from src.types import ClusterId, Geocode

logger = logging.getLogger(__name__)


class GeocodeClusterSchema(dy.Schema):
    """Schema for clustering results for a single k value."""

    geocode = dy.UInt64(nullable=False)
    cluster = dy.UInt32(nullable=False)


class GeocodeClusterMultiKSchema(GeocodeClusterSchema):
    """Schema for clustering results across multiple k values."""

    num_clusters = dy.UInt32(nullable=False)


def build_geocode_cluster_df(
    multi_k_df: dy.DataFrame[GeocodeClusterMultiKSchema],
    num_clusters: int,
) -> dy.DataFrame[GeocodeClusterSchema]:
    """Extract clustering results for a single k value from multi-k results.

    Args:
        multi_k_df: DataFrame containing clustering results for multiple k values
        num_clusters: The specific k value to extract

    Returns:
        DataFrame with clustering results for the specified k value only
    """
    df = multi_k_df.filter(pl.col("num_clusters") == num_clusters).select(
        ["geocode", "cluster"]
    )
    return GeocodeClusterSchema.validate(df)


def build_geocode_cluster_multi_k_df(
    geocode_lf: dy.LazyFrame[GeocodeNoEdgesSchema],
    distance_matrix: GeocodeDistanceMatrix,
    connectivity_matrix: GeocodeConnectivityMatrix,
    min_k: int,
    max_k: int,
) -> dy.DataFrame[GeocodeClusterMultiKSchema]:
    """Build clustering results for all k values from min_k to max_k.

    Args:
        geocode_lf: LazyFrame containing geocode information
        distance_matrix: Precomputed distance matrix between geocodes
        connectivity_matrix: Spatial connectivity constraints for clustering
        min_k: Minimum number of clusters to test
        max_k: Maximum number of clusters to test

    Returns:
        DataFrame with clustering results for all k values tested
    """
    if min_k < 2:
        raise ValueError(f"min_k must be at least 2, got {min_k}")
    if max_k < min_k:
        raise ValueError(f"max_k ({max_k}) must be >= min_k ({min_k})")

    # Collect the LazyFrame once at the start
    geocode_df = geocode_lf.collect()
    geocodes = geocode_df["geocode"]
    num_geocodes = len(geocodes)

    # Validate k range
    if max_k >= num_geocodes:
        logger.warning(
            f"max_k ({max_k}) is >= number of geocodes ({num_geocodes}). "
            f"Reducing max_k to {num_geocodes - 1}"
        )
        max_k = num_geocodes - 1

    logger.info(
        f"Testing {max_k - min_k + 1} cluster configurations (k={min_k} to k={max_k})"
    )

    all_results = []

    for k in range(min_k, max_k + 1):
        logger.info(f"Testing k={k}...")

        clusters = AgglomerativeClustering(
            n_clusters=k,
            connectivity=csr_matrix(connectivity_matrix._connectivity_matrix),  # type: ignore
            linkage="ward",
        ).fit_predict(distance_matrix.squareform())

        assert len(geocodes) == len(clusters)

        k_df = pl.DataFrame(
            data={
                "geocode": geocodes,
                "num_clusters": [k] * len(geocodes),
                "cluster": clusters,
            }
        ).with_columns(
            pl.col("num_clusters").cast(pl.UInt32),
            pl.col("cluster").cast(pl.UInt32),
        )

        all_results.append(k_df)

    df = pl.concat(all_results)
    return GeocodeClusterMultiKSchema.validate(df)


def cluster_ids(
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
) -> List[ClusterId]:
    return geocode_cluster_df["cluster"].unique().to_list()


def iter_clusters_and_geocodes(
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
) -> Iterator[Tuple[ClusterId, List[str]]]:
    for row in (geocode_cluster_df.group_by("cluster").all().sort("cluster")).iter_rows(
        named=True
    ):
        yield row["cluster"], row["geocode"]


def cluster_for_geocode(
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema], geocode: Geocode
) -> ClusterId:
    return geocode_cluster_df.filter(pl.col("geocode") == geocode)["cluster"].to_list()[
        0
    ]


def geocodes_for_cluster(
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema], cluster: ClusterId
) -> List[Geocode]:
    return geocode_cluster_df.filter(pl.col("cluster") == cluster)["geocode"].to_list()


def num_clusters(
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
) -> int:
    num = geocode_cluster_df["cluster"].max()
    assert isinstance(num, int)
    return num
