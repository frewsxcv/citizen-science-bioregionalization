# TODO: Don't include geohashes that extend beyond the bounds of the dataset as those clusters will have artificially fewer counts

from collections import defaultdict, Counter
import pandas as pd
import logging
import numpy as np
import geojson  # type: ignore
import polars as pl
import random
import pickle
import os
import geohashr
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

    taxon: pl.LazyFrame
    """
    Schema:
    - `taxonKey`: `int`
    - `len`: `int`
    - `average`: `float`
    """

    order_counts: pl.LazyFrame
    """
    Schema:
    - `order`: `str`
    - `count`: `int`
    """


def build_geohash_series(dataframe: pl.DataFrame, precision: int) -> pl.Series:
    return (
        dataframe[["decimalLatitude", "decimalLongitude"]]
        .map_rows(lambda n: (geohashr.encode(n[0], n[1], precision)))
        .rename({"map": "geohash"})["geohash"]
    )


class ReadRowsResult(NamedTuple):
    taxon_counts: pl.LazyFrame
    """
    Schema:
    - `geohash`: `str`
    - `taxonKey`: `int`
    - `len`: `int`
    """

    order_counts_series: pl.LazyFrame
    """
    Schema:
    - `geohash`: `str`
    - `order`: `str`
    - `count`: `int`
    """

    taxon_index: pl.LazyFrame
    """
    Schema:
    - `taxonKey`: `int`
    - `verbatimScientificName`: `str`
    """

    @classmethod
    def build(cls, input_file: str, geohash_precision: int) -> Self:
        taxon_counts = pl.LazyFrame(
            schema={
                "geohash": pl.String,
                "taxonKey": pl.UInt64,
                # TODO: rename this to `count`
                "len": pl.UInt32,
            }
        )

        order_counts = pl.LazyFrame(
            schema={
                "geohash": pl.String,
                "order": pl.String,
                "count": pl.UInt32,
            }
        )

        # Will this work for eBird?
        # geohash_to_taxon_id_to_user_to_count: DefaultDict[
        #     Geohash, DefaultDict[TaxonId, Counter[str]]
        # ] = defaultdict(lambda: defaultdict(Counter))

        taxon_index = pl.LazyFrame(
            schema={
                "taxonKey": pl.UInt64,
                "verbatimScientificName": pl.String,
            }
        )

        with Timer(output=logger.info, prefix="Reading rows"):
            for read_dataframe in read_rows(input_file):
                geohash_series = build_geohash_series(read_dataframe, geohash_precision)
                dataframe_with_geohash = read_dataframe.lazy().with_columns(
                    geohash_series
                )

                taxon_counts = pl.concat(
                    items=[
                        taxon_counts,
                        dataframe_with_geohash.group_by(["geohash", "taxonKey"]).agg(
                            pl.len()
                        ),
                    ]
                )

                order_counts = pl.concat(
                    items=[
                        order_counts,
                        dataframe_with_geohash.group_by(["geohash", "order"]).agg(
                            pl.len().alias("count")
                        ),
                    ]
                )

                taxon_index = pl.concat(
                    items=[
                        taxon_index,
                        dataframe_with_geohash.select(
                            ["taxonKey", "verbatimScientificName"]
                        ),
                    ]
                ).unique()

                # for row in dataframe_with_geohash.collect(streaming=True).iter_rows(
                #     named=True
                # ):
                #     geohash_to_taxon_id_to_user_to_count[row["geohash"]][
                #         row["taxonKey"]
                #     ][row["recordedBy"]] += 1
                #     # If the observer has seen the taxon more than 5 times, skip it
                #     if (
                #         geohash_to_taxon_id_to_user_to_count[row["geohash"]][
                #             row["taxonKey"]
                #         ][row["recordedBy"]]
                #         > 5
                #     ):
                #         continue

        taxon_counts = (
            taxon_counts.group_by(["geohash", "taxonKey"])
            .agg(pl.col("len").sum())
            .sort(by="geohash")
        )

        order_counts_series = (
            order_counts.group_by(["geohash", "order"])
            .agg(pl.col("count").sum())
            .lazy()
        )

        return cls(
            taxon_counts=taxon_counts,
            order_counts_series=order_counts_series,
            taxon_index=taxon_index,
        )

    def scientific_name_for_taxon_key(self, taxon_key: TaxonId) -> str:
        column = (
            self.taxon_index.filter(pl.col("taxonKey") == taxon_key)
            .collect()
            .get_column("verbatimScientificName")
        )
        if len(column) > 1:
            # TODO: what should we do here? e.g. "Sciurus carolinensis leucotis" and "Sciurus carolinensis"
            # raise ValueError(f"Multiple scientific names for taxon key {taxon_key}")
            logger.error(f"Multiple scientific names for taxon key {taxon_key}")
            return column.limit(1).item()
        return column.item()

    def ordered_geohashes(self) -> List[Geohash]:
        return (
            self.taxon_counts.select("geohash")
            .unique()
            .sort(by="geohash")
            .collect()
            .get_column("geohash")
            .to_list()
        )

    def ordered_taxon_keys(self) -> List[TaxonId]:
        return (
            self.taxon_counts.select("taxonKey")
            .unique()
            .sort(by="taxonKey")
            .collect()
            .get_column("taxonKey")
            .to_list()
        )

    def build_stats(
        self,
        geohash_filter: Optional[List[Geohash]] = None,
    ) -> Stats:
        geohashes = (
            self.ordered_geohashes()
            if geohash_filter is None
            else [g for g in self.ordered_geohashes() if g in geohash_filter]
        )

        # Schema:
        # - `taxonKey`: `int`
        # - `count`: `int`
        taxon_counts: pl.LazyFrame = (
            self.taxon_counts.filter(pl.col("geohash").is_in(geohashes))
            .select(["taxonKey", "len"])
            .group_by("taxonKey")
            .agg(pl.col("len").sum())
        )

        # Total observation count all filtered geohashes
        total_count: int = taxon_counts.select("len").sum().collect()["len"].item()

        # Schema:
        # - `taxonKey`: `int`
        # - `count`: `int`
        # - `average`: `float`
        taxon: pl.LazyFrame = taxon_counts.with_columns(
            (pl.col("len") / total_count).alias("average")
        )

        order_counts = (
            self.order_counts_series.filter(pl.col("geohash").is_in(geohashes))
            .group_by("order")
            .agg(pl.col("count").sum())
        )

        return Stats(
            geohashes=geohashes,
            taxon=taxon,
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
        #         read_rows_result.taxon_counts.filter(pl.col("geohash") == geohash)
        #         .collect()
        #         .iter_rows(named=False)
        #     ):
        #         j = ordered_seen_taxon_id.index(taxonKey)
        #         matrix[i, j] = np.uint32(count)

        X = read_rows_result.taxon_counts.collect().pivot(
            on="taxonKey",
            index="geohash",
        )

    # fill null values with 0
    with Timer(output=logger.info, prefix="Filling null values"):
        X = X.fill_null(np.uint32(0))

    assert X["geohash"].to_list() == read_rows_result.ordered_geohashes()

    X = X.drop("geohash")

    # filtered.group_by("geohash").agg(pl.col("len").filter(on == value).sum().alias(str(value)) for value in set(taxonKeys)).collect()

    # logger.info(
    #     f"Running pdist on matrix: {len(X.index)} geohashes, {len(X.columns)} taxon IDs"
    # )

    # whitened = whiten(matrix)
    with Timer(output=logger.info, prefix="Running pdist"):
        result = pdist(X.to_numpy(), metric="braycurtis")

    return read_rows_result.ordered_geohashes(), result


def print_cluster_stats(
    cluster: ClusterId,
    geohashes: List[Geohash],
    read_rows_result: ReadRowsResult,
    all_stats: Stats,
) -> None:
    stats = read_rows_result.build_stats(geohash_filter=geohashes)
    print("-" * 10)
    print(f"cluster {cluster} (count: {len(geohashes)})")
    print(
        f"Passeriformes counts: {stats.order_counts.filter(pl.col('order') == 'Passeriformes').collect().get_column('count').item()}"
    )
    print(
        f"Anseriformes counts: {stats.order_counts.filter(pl.col('order') == 'Anseriformes').collect().get_column('count').item()}"
    )

    for taxon_id, count in (
        stats.taxon.sort(by="len", descending=True)
        .limit(5)
        .select(["taxonKey", "len"])
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
            print(f"{read_rows_result.scientific_name_for_taxon_key(taxon_id)}:")
            print(
                f"  - Percentage difference: {'+' if percent_diff > 0 else ''}{percent_diff:.2f}%"
            )
            print(f"  - Proportion: {average * 100:.2f}%")
            print(f"  - Count: {count}")


def print_all_cluster_stats(read_rows_result: ReadRowsResult, all_stats: Stats) -> None:
    for taxon_id, count in (
        all_stats.taxon.sort(by="len", descending=True)
        .limit(5)
        .select(["taxonKey", "len"])
        .collect()
        .iter_rows(named=False)
    ):
        average = (
            all_stats.taxon.filter(pl.col("taxonKey") == taxon_id)
            .collect()
            .get_column("average")
            .item()
        )
        print(f"{read_rows_result.scientific_name_for_taxon_key(taxon_id)}:")
        print(f"  - Proportion: {average * 100:.2f}%")
        print(f"  - Count: {count}")


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

    clusters = list(map(int, fcluster(Z, t=5, criterion="maxclust")))
    logger.info(f"Number of clusters: {len(set(clusters))}")

    cluster_dataframe = ClusterDataFrame.build(ordered_seen_geohash, clusters)
    feature_collection = build_geojson_feature_collection(cluster_dataframe)

    for cluster, geohashes in cluster_dataframe.iter_clusters_and_geohashes():
        print_cluster_stats(cluster, geohashes, read_rows_result, all_stats)

    with open(args.output_file, "w") as geojson_writer:
        geojson.dump(feature_collection, geojson_writer)

    plot_clusters(feature_collection)
