# TODO: Don't include geohashes that extend beyond the bounds of the dataset as those clusters will have artificially fewer counts

from collections import defaultdict, Counter
import pandas as pd
import logging
import numpy as np
import geojson  # type: ignore
import random
import pickle
import os
from contexttimer import Timer
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
    taxon_counts_dataframe: pd.DataFrame
    """
    Schema:
    - `geohash`: `str` (index)
    - `taxon_id`: `str` (index)
    - `count`: `int`
    """

    order_counts_dataframe: pd.DataFrame
    """
    Schema:
    - `geohash`: `str` (index)
    - `order`: `str` (index)
    - `count`: `int`
    """

    ordered_seen_taxon_id: List[TaxonId]
    ordered_seen_geohash: List[Geohash]
    # taxon_id -> scientific_name
    taxon_index: Dict[TaxonId, str]

    @classmethod
    def build(cls, input_file: str, geohash_precision: int) -> Self:
        # { (geohash, taxon_id): count }
        taxon_counts_dataframe_data: Dict[Tuple[Geohash, TaxonId], int] = {}
        order_counts_dataframe_data: Dict[Tuple[Geohash, str], int] = {}
        # Will this work for eBird?
        geohash_to_taxon_id_to_user_to_count: DefaultDict[
            Geohash, DefaultDict[TaxonId, Counter[str]]
        ] = defaultdict(lambda: defaultdict(Counter))

        taxon_index: Dict[TaxonId, str] = {}

        with Timer(output=logger.info, prefix="Reading rows"):
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

                taxon_counts_dataframe_data.setdefault((geohash, row.taxon_id), 0)
                taxon_counts_dataframe_data[(geohash, row.taxon_id)] += 1
                order_counts_dataframe_data.setdefault((geohash, row.order), 0)
                order_counts_dataframe_data[(geohash, row.order)] += 1
                taxon_index[row.taxon_id] = row.scientific_name

        taxon_counts_dataframe: pd.DataFrame = pd.DataFrame(
            data=([count] for count in taxon_counts_dataframe_data.values()),
            columns=["count"],
            index=pd.MultiIndex.from_tuples(
                taxon_counts_dataframe_data.keys(),
                names=["geohash", "taxon_id"],
            ),
        )

        order_counts_dataframe: pd.DataFrame = pd.DataFrame(
            data=([count] for count in order_counts_dataframe_data.values()),
            columns=["count"],
            index=pd.MultiIndex.from_tuples(
                order_counts_dataframe_data.keys(),
                names=["geohash", "order"],
            ),
        )

        ordered_seen_taxon_id = sorted(
            taxon_counts_dataframe.index.get_level_values("taxon_id")
        )
        ordered_seen_geohash = sorted(
            taxon_counts_dataframe.index.get_level_values("geohash")
        )
        return cls(
            taxon_counts_dataframe=taxon_counts_dataframe,
            order_counts_dataframe=order_counts_dataframe,
            ordered_seen_taxon_id=ordered_seen_taxon_id,
            ordered_seen_geohash=ordered_seen_geohash,
            taxon_index=taxon_index,
        )

    def build_stats(
        self,
        geohash_filter: Optional[List[Geohash]] = None,
    ) -> Stats:
        # taxon_id -> taxon average
        averages: DefaultDict[TaxonId, float] = defaultdict(float)
        # taxon_id -> taxon count
        counts: Counter[TaxonId] = Counter()
        # taxonomic order -> count
        order_counts: Counter[str] = Counter()

        geohashes = (
            list(self.taxon_counts_dataframe.index.get_level_values("geohash"))
            if geohash_filter is None
            else [
                g
                for g in self.taxon_counts_dataframe.index.get_level_values("geohash")
                if g in geohash_filter
            ]
        )

        # Calculate total counts for each taxon_id using the dataframe
        filtered_df = (
            self.taxon_counts_dataframe
            if geohash_filter is None
            else self.taxon_counts_dataframe.loc[geohash_filter]
        )
        for (_, taxon_id), row in filtered_df.iterrows():
            counts[taxon_id] += row["count"]

        filtered_order_counts_dataframe = (
            self.order_counts_dataframe
            if geohash_filter is None
            else self.order_counts_dataframe.loc[geohash_filter]
        )
        for (_, order), row in filtered_order_counts_dataframe.iterrows():
            order_counts[order] += row["count"]

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
    # Create a matrix where each row is a geohash and each column is a taxon ID
    # Example:
    # [
    #     [1, 0, 0, 0],  # geohash 1 has 1 occurrence of taxon 1, 0 occurrences of taxon 2, 0 occurrences of taxon 3, 0 occurrences of taxon 4
    #     [0, 2, 0, 1],  # geohash 2 has 0 occurrences of taxon 1, 2 occurrences of taxon 2, 0 occurrences of taxon 3, 1 occurrences of taxon 4
    #     [0, 0, 3, 0],  # geohash 3 has 0 occurrences of taxon 1, 0 occurrences of taxon 2, 3 occurrences of taxon 3, 0 occurrences of taxon 4
    #     [0, 2, 0, 4],  # geohash 4 has 0 occurrences of taxon 1, 2 occurrences of taxon 2, 0 occurrences of taxon 3, 4 occurrences of taxon 4
    # ]
    with Timer(output=logger.info, prefix="Building matrix"):
        X = read_rows_result.taxon_counts_dataframe.unstack(fill_value=0)

    logger.info(
        f"Running pdist on matrix: {len(X.index)} geohashes, {len(X.columns)} taxon IDs"
    )

    # whitened = whiten(matrix)
    with Timer(output=logger.info, prefix="Running pdist"):
        result = pdist(X.values, metric="braycurtis")

    return list(X.index), result


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
    logging.basicConfig(filename=args.log_file, encoding="utf-8", level=logging.INFO)

    if os.path.exists("condensed_distance_matrix.pickle"):
        with Timer(output=logger.info, prefix="Loading condensed distance matrix"):
            with open("condensed_distance_matrix.pickle", "rb") as pickle_reader:
                ordered_seen_geohash, condensed_distance_matrix, read_rows_result = (
                    pickle.load(pickle_reader)
                )
    else:
        read_rows_result = ReadRowsResult.build(input_file, args.geohash_precision)
        ordered_seen_geohash, condensed_distance_matrix = (
            build_condensed_distance_matrix(read_rows_result)
        )
        with Timer(output=logger.info, prefix="Saving condensed distance matrix"):
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
