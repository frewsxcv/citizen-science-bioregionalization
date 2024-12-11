from collections import defaultdict, Counter
import pandas as pd
import logging
import numpy as np
import geojson  # type: ignore
import random
import pickle
import os
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from typing import (
    DefaultDict,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Self,
    Set,
    Tuple,
)

from src.cli import parse_arguments
from src.darwin_core import TaxonId, read_rows
from src.geohash import geohash_to_bbox, Geohash
from src.render import plot_clusters
from src.cluster import ClusterId

COLORS = [
    "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
    for _ in range(1000)
]

logger = logging.getLogger(__name__)


def build_geojson_geohash_polygon(geohash: Geohash) -> geojson.Polygon:
    bbox = geohash_to_bbox(geohash)
    return geojson.Polygon(
        coordinates=[
            [
                [bbox.sw.lon, bbox.sw.lat],
                [bbox.ne.lon, bbox.sw.lat],
                [bbox.ne.lon, bbox.ne.lat],
                [bbox.sw.lon, bbox.ne.lat],
                [bbox.sw.lon, bbox.sw.lat],
            ]
        ]
    )


def build_geojson_feature(
    geohashes: List[Geohash], cluster: ClusterId
) -> geojson.Feature:
    geometries = [build_geojson_geohash_polygon(geohash) for geohash in geohashes]
    geometry = (
        geojson.GeometryCollection(geometries) if len(geometries) > 1 else geometries[0]
    )

    return geojson.Feature(
        properties={
            "label": ", ".join(geohashes),
            "fill": COLORS[cluster],
            "stroke-width": 0,
            "cluster": cluster,
        },
        geometry=geometry,
    )


class Stats(NamedTuple):
    geohashes: List[Geohash]
    # taxon_id -> average per geohash
    averages: Dict[TaxonId, float]
    # taxon_id -> count
    counts: Counter[TaxonId]
    # taxonomic order -> count
    order_counts: Counter[str]


class ReadRowsResult(NamedTuple):
    geohash_to_taxon_id_to_count: DefaultDict[Geohash, Counter[TaxonId]]
    geohash_to_order_to_count: DefaultDict[Geohash, Counter[str]]
    seen_taxon_id: Set[TaxonId]
    ordered_seen_taxon_id: List[TaxonId]
    ordered_seen_geohash: List[Geohash]
    # taxon_id -> scientific_name
    taxon_index: Dict[TaxonId, str]

    @classmethod
    def build(cls, input_file: str, geohash_precision: int) -> Self:
        geohash_to_taxon_id_to_count: DefaultDict[Geohash, Counter[TaxonId]] = (
            defaultdict(Counter)
        )
        geohash_to_order_to_count: DefaultDict[Geohash, Counter[str]] = defaultdict(
            Counter
        )
        seen_taxon_id: Set[TaxonId] = set()
        # Will this work for eBird?
        geohash_to_taxon_id_to_user_to_count: DefaultDict[
            Geohash, DefaultDict[TaxonId, Counter[str]]
        ] = defaultdict(lambda: defaultdict(Counter))

        logger.info("Reading rows")
        taxon_index: Dict[TaxonId, str] = {}

        for row in read_rows(input_file):
            geohash = row.geohash(geohash_precision)
            geohash_to_taxon_id_to_user_to_count[geohash][row.taxon_id][
                row.observer
            ] += 1
            # If the observer has seen the taxon more than 5 times, skip it
            if (
                geohash_to_taxon_id_to_user_to_count[geohash][row.taxon_id][
                    row.observer
                ]
                > 5
            ):
                continue
            geohash_to_taxon_id_to_count[geohash][row.taxon_id] += 1
            geohash_to_order_to_count[geohash][row.order] += 1
            taxon_index[row.taxon_id] = row.scientific_name
            seen_taxon_id.add(row.taxon_id)

        ordered_seen_taxon_id = sorted(seen_taxon_id)
        ordered_seen_geohash = sorted(geohash_to_taxon_id_to_count.keys())
        return cls(
            geohash_to_taxon_id_to_count,
            geohash_to_order_to_count,
            seen_taxon_id,
            ordered_seen_taxon_id,
            ordered_seen_geohash,
            taxon_index,
        )

    def build_stats(self, geohash_filter: Optional[List[Geohash]] = None) -> Stats:
        # taxon_id -> taxon average
        averages: DefaultDict[TaxonId, float] = defaultdict(float)
        # taxon_id -> taxon count
        counts: Counter[TaxonId] = Counter()
        # taxonomic order -> count
        order_counts: Counter[str] = Counter()

        geohashes = (
            list(self.geohash_to_taxon_id_to_count.keys())
            if geohash_filter is None
            else [
                g
                for g in self.geohash_to_taxon_id_to_count.keys()
                if g in geohash_filter
            ]
        )

        # Calculate total counts for each taxon_id
        for (
            geohash,
            taxon_counts,
        ) in read_rows_result.geohash_to_taxon_id_to_count.items():
            if geohash_filter is not None and geohash not in geohash_filter:
                continue
            for taxon_id, count in taxon_counts.items():
                counts[taxon_id] += count

        for (
            geohash,
            inner_order_counts,
        ) in read_rows_result.geohash_to_order_to_count.items():
            if geohash_filter is not None and geohash not in geohash_filter:
                continue
            for order, count in inner_order_counts.items():
                order_counts[order] += count

        # Calculate averages for each taxon_id
        for taxon_id in counts.keys():
            averages[taxon_id] = counts[taxon_id] / sum(counts.values())

        return Stats(
            geohashes=geohashes,
            averages=averages,
            counts=counts,
            order_counts=order_counts,
        )


def build_condensed_distance_matrix(
    read_rows_result: ReadRowsResult,
) -> Tuple[List[str], np.ndarray]:
    ordered_seen_taxon_id = sorted(read_rows_result.seen_taxon_id)
    ordered_seen_geohash = sorted(read_rows_result.geohash_to_taxon_id_to_count.keys())

    logger.info(
        f"Building condensed distance matrix: {len(ordered_seen_geohash)} geohashes, {len(ordered_seen_taxon_id)} taxon IDs"
    )

    # Create a matrix where each row is a geohash and each column is a taxon ID
    # Example:
    # [
    #     [1, 0, 0, 0],  # geohash 1 has 1 occurrence of taxon 1, 0 occurrences of taxon 2, 0 occurrences of taxon 3, 0 occurrences of taxon 4
    #     [0, 2, 0, 1],  # geohash 2 has 0 occurrences of taxon 1, 2 occurrences of taxon 2, 0 occurrences of taxon 3, 1 occurrences of taxon 4
    #     [0, 0, 3, 0],  # geohash 3 has 0 occurrences of taxon 1, 0 occurrences of taxon 2, 3 occurrences of taxon 3, 0 occurrences of taxon 4
    #     [0, 2, 0, 4],  # geohash 4 has 0 occurrences of taxon 1, 2 occurrences of taxon 2, 0 occurrences of taxon 3, 4 occurrences of taxon 4
    # ]
    matrix = np.zeros((len(ordered_seen_geohash), len(ordered_seen_taxon_id)))
    for i, geohash in enumerate(ordered_seen_geohash):
        for j, taxon_id in enumerate(ordered_seen_taxon_id):
            matrix[i, j] = read_rows_result.geohash_to_taxon_id_to_count[geohash].get(
                taxon_id, 0
            )

    # whitened = whiten(matrix)

    return ordered_seen_geohash, pdist(matrix, metric="braycurtis")


def print_cluster_stats(
    cluster: ClusterId,
    geohashes: List[Geohash],
    read_rows_result: ReadRowsResult,
    all_stats: Stats,
) -> None:
    stats = read_rows_result.build_stats(geohash_filter=geohashes)
    print("-" * 10)
    print(f"cluster {cluster} (count: {len(geohashes)})")
    print(f"Passeriformes counts: {stats.order_counts.get('Passeriformes')}")
    print(f"Anseriformes counts: {stats.order_counts.get('Anseriformes')}")
    for taxon_id, count in sorted(
        all_stats.counts.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        # If the difference between the average of the cluster and the average of all is greater than 20%, print it
        percent_diff = (
            stats.averages[taxon_id] / all_stats.averages[taxon_id] * 100
        ) - 100
        if abs(percent_diff) > 20:
            # Print the percentage difference
            print(
                f"{read_rows_result.taxon_index[taxon_id]}: {stats.averages[taxon_id]} {all_stats.averages[taxon_id]} ({percent_diff:.2f}%)"
            )


def print_all_cluster_stats(read_rows_result: ReadRowsResult, all_stats: Stats) -> None:
    print("all")
    for taxon_id, count in sorted(
        all_stats.counts.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(
            f"{read_rows_result.taxon_index[taxon_id]}: {all_stats.averages[taxon_id]}"
        )


class ClusterDataFrame(NamedTuple):
    dataframe: pd.DataFrame
    """
    Schema:
    - `geohash`: `str` (index)
    - `cluster`: `int`
    """

    @classmethod
    def build(
        cls, ordered_seen_geohash: List[Geohash], clusters: List[ClusterId]
    ) -> Self:
        dataframe = pd.DataFrame(
            columns=["geohash", "cluster"],
            data=(
                (geohash, cluster)
                for geohash, cluster in zip(ordered_seen_geohash, clusters)
            ),
        )
        dataframe.set_index("geohash", inplace=True)
        return cls(dataframe)

    def iter_clusters_and_geohashes(
        self,
    ) -> Iterator[Tuple[ClusterId, List[Geohash]]]:
        for cluster, geohashes in self.dataframe.groupby("cluster"):
            yield cluster, geohashes.index


def build_geojson_feature_collection(
    cluster_dataframe: ClusterDataFrame,
) -> geojson.FeatureCollection:
    return geojson.FeatureCollection(
        features=[
            build_geojson_feature(geohashes, cluster)
            for cluster, geohashes in cluster_dataframe.iter_clusters_and_geohashes()
        ],
    )


if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file
    logging.basicConfig(filename=args.log_file, encoding="utf-8", level=logging.DEBUG)

    if os.path.exists("condensed_distance_matrix.pickle"):
        logger.info("Loading condensed distance matrix")
        with open("condensed_distance_matrix.pickle", "rb") as pickle_reader:
            ordered_seen_geohash, condensed_distance_matrix, read_rows_result = (
                pickle.load(pickle_reader)
            )
    else:
        read_rows_result = ReadRowsResult.build(input_file, args.geohash_precision)
        ordered_seen_geohash, condensed_distance_matrix = (
            build_condensed_distance_matrix(read_rows_result)
        )
        logger.info("Saving condensed distance matrix")
        with open("condensed_distance_matrix.pickle", "wb") as pickle_writer:
            pickle.dump(
                (ordered_seen_geohash, condensed_distance_matrix, read_rows_result),
                pickle_writer,
            )

    # Find the top averages of taxon
    all_stats = read_rows_result.build_stats()
    # For each top count taxon, print the average per geohash
    print_all_cluster_stats(read_rows_result, all_stats)

    # Generate the linkage matrix
    Z = linkage(condensed_distance_matrix, "ward")
    # fig = plt.figure(figsize=(25, 10))
    # dn = dendrogram(Z, labels=ordered_seen_geohash)
    # plt.show()

    clusters = list(map(int, fcluster(Z, t=10, criterion="maxclust")))
    logger.info(f"Number of clusters: {len(set(clusters))}")

    cluster_dataframe = ClusterDataFrame.build(ordered_seen_geohash, clusters)
    feature_collection = build_geojson_feature_collection(cluster_dataframe)

    for cluster, geohashes in cluster_dataframe.iter_clusters_and_geohashes():
        print_cluster_stats(cluster, geohashes, read_rows_result, all_stats)

    with open(args.output_file, "w") as geojson_writer:
        geojson.dump(feature_collection, geojson_writer)

    plot_clusters(feature_collection)
