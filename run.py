# TODO: Don't include geohashes that extend beyond the bounds of the dataset
# so those clusters will have artificially fewer counts

import random
import logging
import numpy as np
import geojson  # type: ignore
import polars as pl
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from contexttimer import Timer
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from typing import (
    Iterator,
    List,
    NamedTuple,
    Optional,
    Self,
    Tuple,
)

from src.cli import parse_arguments
from src.cluster_color_builder import ClusterColorBuilder
from src.cluster_stats import Stats
from src.darwin_core_aggregations import DarwinCoreAggregations
from src.geohash import Geohash
from src.render import plot_clusters
from src.cluster import ClusterId
import matplotlib.pyplot as plt
from src.geojson import build_geojson_feature_collection
import os

logger = logging.getLogger(__name__)


def build_condensed_distance_matrix(
    darwin_core_aggregations: DarwinCoreAggregations,
) -> np.ndarray:
    cache_file = "condensed_distance_matrix.parquet"

    # Try to load from cache
    if os.path.exists(cache_file):
        with Timer(output=logger.info, prefix="Loading cached distance matrix"):
            matrix_df = pl.read_parquet(cache_file)
            return matrix_df.to_numpy().flatten()

    # Create a matrix where each row is a geohash and each column is a taxon ID
    # Example:
    # [
    #     [1, 0, 0, 0],  # geohash 1 has 1 occurrence of taxon 1, 0 occurrences of taxon 2, 0 occurrences of taxon 3, 0 occurrences of taxon 4
    #     [0, 2, 0, 1],  # geohash 2 has 0 occurrences of taxon 1, 2 occurrences of taxon 2, 0 occurrences of taxon 3, 1 occurrences of taxon 4
    #     [0, 0, 3, 0],  # geohash 3 has 0 occurrences of taxon 1, 0 occurrences of taxon 2, 3 occurrences of taxon 3, 0 occurrences of taxon 4
    #     [0, 2, 0, 4],  # geohash 4 has 0 occurrences of taxon 1, 2 occurrences of taxon 2, 0 occurrences of taxon 3, 4 occurrences of taxon 4
    # ]
    with Timer(output=logger.info, prefix="Building matrix"):
        # matrix = np.zeros(
        #     (len(ordered_seen_geohash), len(ordered_seen_taxon_id)), dtype=np.uint32
        # )
        # geohash_count = 0
        # for i, geohash in enumerate(ordered_seen_geohash):
        #     # Print progress every 1000 geohashes
        #     if geohash_count % 1000 == 0:
        #         logger.info(
        #             f"Processing geohash {geohash_count} of {len(ordered_seen_geohash)} ({geohash_count / len(ordered_seen_geohash) * 100:.2f}%)"
        #         )
        #     geohash_count += 1

        #     for geohash, taxonKey, count in (
        #         darwin_core_aggregations.taxon_counts.filter(pl.col("geohash") == geohash)
        #         .collect()
        #         .iter_rows(named=False)
        #     ):
        #         j = ordered_seen_taxon_id.index(taxonKey)
        #         matrix[i, j] = np.uint32(count)

        X = darwin_core_aggregations.taxon_counts.pivot(
            on="taxonKey",
            index="geohash",
        )

    assert X.height > 1, "More than one geohash is required to cluster"

    # fill null values with 0
    with Timer(output=logger.info, prefix="Filling null values"):
        X = X.fill_null(np.uint32(0))

    assert X["geohash"].to_list() == darwin_core_aggregations.ordered_geohashes()

    with Timer(output=logger.info, prefix="Dropping geohash column"):
        X = X.drop("geohash")

    # filtered.group_by("geohash").agg(pl.col("len").filter(on == value).sum().alias(str(value)) for value in set(taxonKeys)).collect()

    with Timer(output=logger.info, prefix="Scaling values"):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    logger.info(f"Reducing dimensions with PCA. Previously: {X.shape}")

    pca = IncrementalPCA(n_components=3000, copy=True, batch_size=3000)

    # Use PCA to reduce the number of dimensions
    with Timer(output=logger.info, prefix="Fitting PCA"):
        X = pca.fit_transform(X)

    logger.info(
        f"Reduced dimensions with PCA. Now: {X.shape[0]} geohashes, {X.shape[1]} taxon IDs"
    )

    logger.info(
        f"Running pdist on matrix: {X.shape[0]} geohashes, {X.shape[1]} taxon IDs"
    )

    with Timer(output=logger.info, prefix="Running pdist"):
        Y = pdist(X, metric="braycurtis")

    with Timer(output=logger.info, prefix="Caching distance matrix"):
        # Convert the condensed distance matrix to a DataFrame and save
        matrix_df = pl.DataFrame({"values": Y})
        matrix_df.write_parquet(cache_file)

    return Y


def print_cluster_stats(
    cluster: ClusterId,
    geohashes: List[Geohash],
    darwin_core_aggregations: DarwinCoreAggregations,
    all_stats: Stats,
) -> None:
    stats = Stats.build(darwin_core_aggregations, geohash_filter=geohashes)
    print("-" * 10)
    print(f"cluster {cluster} (count: {len(geohashes)})")

    for taxon_id, count in (
        stats.taxon.sort(by="count", descending=True)
        .limit(5)
        .select(["taxonKey", "count"])
        .collect()
        .iter_rows(named=False)
    ):
        average = (
            stats.taxon.filter(pl.col("taxonKey") == taxon_id)
            .collect()
            .get_column("average")
            .item()
        )
        all_average = (
            all_stats.taxon.filter(pl.col("taxonKey") == taxon_id)
            .collect()
            .get_column("average")
            .item()
        )

        # If the difference between the average of the cluster and the average of all is greater than 20%, print it
        percent_diff = (average / all_average * 100) - 100
        if abs(percent_diff) > 20:
            # Print the percentage difference
            print(
                f"{darwin_core_aggregations.scientific_name_for_taxon_key(taxon_id)}:"
            )
            print(
                f"  - Percentage difference: {'+' if percent_diff > 0 else ''}{percent_diff:.2f}%"
            )
            print(f"  - Proportion: {average * 100:.2f}%")
            print(f"  - Count: {count}")


def print_all_cluster_stats(
    darwin_core_aggregations: DarwinCoreAggregations, all_stats: Stats
) -> None:
    for taxon_id, count in (
        all_stats.taxon.sort(by="count", descending=True)
        .limit(5)
        .select(["taxonKey", "count"])
        .collect()
        .iter_rows(named=False)
    ):
        average = (
            all_stats.taxon.filter(pl.col("taxonKey") == taxon_id)
            .collect()
            .get_column("average")
            .item()
        )
        print(f"{darwin_core_aggregations.scientific_name_for_taxon_key(taxon_id)}:")
        print(f"  - Proportion: {average * 100:.2f}%")
        print(f"  - Count: {count}")


def show_dendrogram(Z: np.ndarray, ordered_seen_geohash: List[Geohash]) -> None:
    plt.figure()
    dendrogram(Z, labels=ordered_seen_geohash)
    plt.show()


COLORS = [
    "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
    for _ in range(1000)
]


class ClusterIndex(NamedTuple):
    dataframe: pl.DataFrame
    """
    Schema:
    - `geohash`: `str`
    - `cluster`: `int`
    """

    # cluster_colors: pl.DataFrame
    # """
    # Schema:
    # - `cluster`: `int`
    # - `color`: `str`
    # """

    @classmethod
    def build(
        cls, ordered_seen_geohash: List[Geohash], clusters: List[ClusterId]
    ) -> Self:
        dataframe = pl.DataFrame(
            data={
                "geohash": ordered_seen_geohash,
                "cluster": clusters,
            },
            schema={"geohash": pl.String, "cluster": pl.UInt32},
        )
        # cluster_colors = pl.DataFrame(
        #     data={
        #         "cluster": list(range(1, max(clusters) + 1)),
        #         "color": [COLORS[i] for i in range(max(clusters))],
        #     },
        #     schema={"cluster": pl.UInt32, "color": pl.String},
        # )
        return cls(
            dataframe,
            # cluster_colors,
        )

    def geohashes_for_cluster(self, cluster: ClusterId) -> List[Geohash]:
        return self.dataframe.filter(pl.col("cluster") == cluster)["geohash"].to_list()

    def determine_color_for_cluster(
        self, cluster: ClusterId, darwin_core_aggregations: DarwinCoreAggregations
    ) -> str:
        stats = Stats.build(
            darwin_core_aggregations,
            geohash_filter=self.geohashes_for_cluster(cluster),
        )
        return ClusterColorBuilder.determine_color_for_cluster(stats)

    def iter_clusters_and_geohashes(
        self,
    ) -> Iterator[Tuple[ClusterId, List[Geohash]]]:
        for row in (self.dataframe.group_by("cluster").all().sort("cluster")).iter_rows(
            named=True
        ):
            yield row["cluster"], row["geohash"]

    def num_clusters(self) -> int:
        num = self.dataframe["cluster"].max()
        assert isinstance(num, int)
        return num


def write_geojson(
    feature_collection: geojson.FeatureCollection, output_file: str
) -> None:
    with open(output_file, "w") as geojson_writer:
        geojson.dump(feature_collection, geojson_writer)


def print_results(
    darwin_core_aggregations: DarwinCoreAggregations,
    all_stats: Stats,
    cluster_index: ClusterIndex,
) -> None:
    # For each top count taxon, print the average per geohash
    print_all_cluster_stats(darwin_core_aggregations, all_stats)

    logger.info(f"Number of clusters: {cluster_index.num_clusters()}")

    for cluster, geohashes in cluster_index.iter_clusters_and_geohashes():
        print_cluster_stats(cluster, geohashes, darwin_core_aggregations, all_stats)


def cluster(
    darwin_core_aggregations: DarwinCoreAggregations,
    num_clusters: int,
    ordered_seen_geohash: List[Geohash],
    show_dendrogram_opt: bool,
) -> ClusterIndex:
    Y = build_condensed_distance_matrix(darwin_core_aggregations)
    Z = linkage(Y, "ward")

    clusters = list(map(int, fcluster(Z, t=num_clusters, criterion="maxclust")))

    if show_dendrogram_opt:
        show_dendrogram(Z, ordered_seen_geohash)

    cluster_index = ClusterIndex.build(ordered_seen_geohash, clusters)

    return cluster_index


def run() -> None:
    args = parse_arguments()
    input_file = args.input_file
    logging.basicConfig(filename=args.log_file, encoding="utf-8", level=logging.INFO)

    darwin_core_aggregations = DarwinCoreAggregations.build(
        input_file, args.geohash_precision
    )
    ordered_seen_geohash = darwin_core_aggregations.ordered_geohashes()

    cluster_dataframe = cluster(
        darwin_core_aggregations,
        args.num_clusters,
        ordered_seen_geohash,
        args.show_dendrogram,
    )

    # Find the top averages of taxon
    all_stats = Stats.build(darwin_core_aggregations)

    feature_collection = build_geojson_feature_collection(
        (
            cluster,
            geohashes,
            cluster_dataframe.determine_color_for_cluster(
                cluster, darwin_core_aggregations
            ),
        )
        for cluster, geohashes in cluster_dataframe.iter_clusters_and_geohashes()
    )

    print_results(darwin_core_aggregations, all_stats, cluster_dataframe)

    if args.plot:
        plot_clusters(feature_collection, num_clusters=args.num_clusters)

    write_geojson(feature_collection, args.output_file)


if __name__ == "__main__":
    run()
