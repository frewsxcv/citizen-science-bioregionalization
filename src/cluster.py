import logging
import os

import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import IncrementalPCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from src import darwin_core_aggregations, cluster_index, dendrogram
from src.logging import log_action
from contexttimer import Timer

logger = logging.getLogger(__name__)


def pivot_taxon_counts(taxon_counts: pl.DataFrame) -> pl.DataFrame:
    """
    Create a matrix where each row is a geohash and each column is a taxon ID

    Example:

    ```
    [
        [1, 0, 0, 0],  # geohash 1 has 1 occurrence of taxon 1, 0 occurrences of taxon 2, 0 occurrences of taxon 3, 0 occurrences of taxon 4
        [0, 2, 0, 1],  # geohash 2 has 0 occurrences of taxon 1, 2 occurrences of taxon 2, 0 occurrences of taxon 3, 1 occurrences of taxon 4
        [0, 0, 3, 0],  # geohash 3 has 0 occurrences of taxon 1, 0 occurrences of taxon 2, 3 occurrences of taxon 3, 0 occurrences of taxon 4
        [0, 2, 0, 4],  # geohash 4 has 0 occurrences of taxon 1, 2 occurrences of taxon 2, 0 occurrences of taxon 3, 4 occurrences of taxon 4
    ]
    ```
    """
    return taxon_counts.pivot(
        on=["kingdom", "species"], index="geohash", values="count"
    )


def build_X(
    darwin_core_aggregations: darwin_core_aggregations.DarwinCoreAggregations,
) -> pl.DataFrame:
    X = log_action(
        "Building matrix",
        lambda: darwin_core_aggregations.taxon_counts.pipe(pivot_taxon_counts),
    )

    assert X.height > 1, "More than one geohash is required to cluster"

    # fill null values with 0
    X = log_action("Filling null values", lambda: X.fill_null(np.uint32(0)))

    assert X["geohash"].to_list() == darwin_core_aggregations.ordered_geohashes()

    X = log_action("Dropping geohash column", lambda: X.drop("geohash"))

    return log_action("Scaling values", lambda: X.pipe(scale_values))


def scale_values(X: pl.DataFrame) -> pl.DataFrame:
    scaler = RobustScaler()
    return pl.from_numpy(scaler.fit_transform(X))


def reduce_dimensions(X: pl.DataFrame) -> pl.DataFrame:
    pca = IncrementalPCA(n_components=3000, copy=True, batch_size=3000)
    return pl.from_numpy(pca.fit_transform(X))


def build_condensed_distance_matrix(
    darwin_core_aggregations: darwin_core_aggregations.DarwinCoreAggregations,
    use_cache: bool,
) -> np.ndarray:
    cache_file = "condensed_distance_matrix.parquet"

    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        with Timer(output=logger.info, prefix="Loading cached distance matrix"):
            matrix_df = pl.read_parquet(cache_file)
            return matrix_df.to_numpy().flatten()

    X = build_X(darwin_core_aggregations)

    # tsne = TSNE(metric="braycurtis")
    # tsne_result = tsne.fit_transform(X)

    # filtered.group_by("geohash").agg(pl.col("len").filter(on == value).sum().alias(str(value)) for value in set(taxonKeys)).collect()

    logger.info(f"Reducing dimensions with PCA. Previously: {X.shape}")

    # Use PCA to reduce the number of dimensions
    X = log_action("Fitting PCA", lambda: X.pipe(reduce_dimensions))

    logger.info(
        f"Reduced dimensions with PCA. Now: {X.shape[0]} geohashes, {X.shape[1]} taxon IDs"
    )

    Y = log_action(
        f"Running pdist on matrix: {X.shape[0]} geohashes, {X.shape[1]} taxon IDs",
        lambda: pdist(X, metric="braycurtis"),
    )

    if use_cache:
        with Timer(output=logger.info, prefix="Caching distance matrix"):
            # Convert the condensed distance matrix to a DataFrame and save
            matrix_df = pl.DataFrame({"values": Y})
            matrix_df.write_parquet(cache_file)

    return Y


def run(
    darwin_core_aggregations: darwin_core_aggregations.DarwinCoreAggregations,
    num_clusters: int,
    show_dendrogram_opt: bool,
    use_cache: bool,
) -> cluster_index.ClusterIndex:
    ordered_seen_geohash = darwin_core_aggregations.ordered_geohashes()
    Y = build_condensed_distance_matrix(darwin_core_aggregations, use_cache)
    Z = linkage(Y, "ward")

    clusters = list(map(int, fcluster(Z, t=num_clusters, criterion="maxclust")))

    if show_dendrogram_opt:
        dendrogram.show(Z, ordered_seen_geohash)

    return cluster_index.build(ordered_seen_geohash, clusters)
