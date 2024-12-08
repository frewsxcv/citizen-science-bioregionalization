from collections import defaultdict, Counter
import csv
import json
import logging
import numpy as np
import pygeohash
import random
import scipy.cluster.hierarchy
import scipy.spatial
import pickle
import os
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.vq import whiten, kmeans2
from scipy.spatial.distance import squareform, pdist
from typing import (
    DefaultDict,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)
import argparse

COLORS = [
    "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
    for _ in range(1000)
]

logger = logging.getLogger(__name__)

type Geohash = str

type TaxonId = int


class Point(NamedTuple):
    lat: float
    lon: float


class Row(NamedTuple):
    location: Point
    taxon_id: TaxonId
    scientific_name: str
    # TODO: Should this be a user ID?
    observer: str

    def geohash(self, precision: int) -> str:
        return pygeohash.encode(
            self.location.lat, self.location.lon, precision=precision
        )


def read_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def read_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster geohash data.")
    parser.add_argument(
        "--geohash-precision",
        type=int,
        help="Precision of the geohash",
        required=True,
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to the log file",
        required=True,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file",
    )
    return parser.parse_args()


def read_rows(input_file: str) -> Generator[Row, None, None]:
    with open(input_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            lat, lon = read_float(row["decimalLatitude"]), read_float(
                row["decimalLongitude"]
            )
            if not (lat and lon):
                logger.error(
                    f"Invalid latitude or longitude: {row['decimalLatitude']}, {row['decimalLongitude']}"
                )
                continue
            taxon_id = read_int(row["taxonKey"])
            if not taxon_id:
                logger.error(f"Invalid taxon ID: {row['taxonKey']}")
                continue
            yield Row(
                Point(lat, lon), taxon_id, row["scientificName"], row["recordedBy"]
            )


class Bbox(NamedTuple):
    sw: Point
    ne: Point


def geohash_to_bbox(geohash: Geohash) -> Bbox:
    lat, lon, lat_err, lon_err = pygeohash.decode_exactly(geohash)
    return Bbox(
        sw=Point(lat=lat - lat_err, lon=lon - lon_err),
        ne=Point(lat=lat + lat_err, lon=lon + lon_err),
    )


def build_geojson_feature(geohashes: List[Geohash], cluster: int) -> Dict:
    geometries = []
    for geohash in geohashes:
        bbox = geohash_to_bbox(geohash)
        coords = [
            [bbox.sw.lon, bbox.sw.lat],
            [bbox.ne.lon, bbox.sw.lat],
            [bbox.ne.lon, bbox.ne.lat],
            [bbox.sw.lon, bbox.ne.lat],
            [bbox.sw.lon, bbox.sw.lat],
        ]
        geometries.append({"type": "Polygon", "coordinates": [coords]})

    return {
        "type": "Feature",
        "properties": {
            "label": ", ".join(geohashes),
            "fill": COLORS[int(cluster)],
            "stroke-width": 0,
            "cluster": int(cluster),
        },
        "geometry": {"type": "GeometryCollection", "geometries": geometries},
    }


class ReadRowsResult(NamedTuple):
    geohash_to_taxon_id_to_count: Dict[Geohash, Counter[TaxonId]]
    seen_taxon_id: Set[TaxonId]
    ordered_seen_taxon_id: List[TaxonId]
    ordered_seen_geohash: List[Geohash]
    # taxon_id -> scientific_name
    taxon_index: Dict[TaxonId, str]


def build_read_rows_result(input_file: str, geohash_precision: int) -> ReadRowsResult:
    geohash_to_taxon_id_to_count: Dict[Geohash, Counter[TaxonId]] = {}
    seen_taxon_id: Set[TaxonId] = set()
    # Will this work for eBird?
    geohash_to_taxon_id_to_user_to_count: Dict[Geohash, Dict[TaxonId, Counter[str]]] = (
        {}
    )

    logger.info("Reading rows")
    taxon_index: Dict[TaxonId, str] = {}

    for row in read_rows(input_file):
        geohash = row.geohash(geohash_precision)
        geohash_to_taxon_id_to_user_to_count.setdefault(geohash, {})
        geohash_to_taxon_id_to_user_to_count[geohash].setdefault(
            row.taxon_id, Counter()
        )
        geohash_to_taxon_id_to_user_to_count[geohash][row.taxon_id][row.observer] += 1
        # If the observer has seen the taxon more than 5 times, skip it
        if (
            geohash_to_taxon_id_to_user_to_count[geohash][row.taxon_id][row.observer]
            > 5
        ):
            continue
        geohash_to_taxon_id_to_count.setdefault(geohash, Counter())
        geohash_to_taxon_id_to_count[geohash][row.taxon_id] += 1
        taxon_index[row.taxon_id] = row.scientific_name
        seen_taxon_id.add(row.taxon_id)

    ordered_seen_taxon_id = sorted(seen_taxon_id)
    ordered_seen_geohash = sorted(geohash_to_taxon_id_to_count.keys())
    return ReadRowsResult(
        geohash_to_taxon_id_to_count,
        seen_taxon_id,
        ordered_seen_taxon_id,
        ordered_seen_geohash,
        taxon_index,
    )


def build_condensed_distance_matrix(
    read_rows_result: ReadRowsResult,
) -> Tuple[List[str], np.ndarray]:
    ordered_seen_taxon_id = sorted(read_rows_result.seen_taxon_id)
    ordered_seen_geohash = sorted(read_rows_result.geohash_to_taxon_id_to_count.keys())

    logger.info("Building condensed distance matrix")

    # Create a matrix where each row is a geohash and each column is a taxon ID
    # Example:
    # [
    #     [1, 0, 0, 0],  # geohash 1 has 1 occurrence of taxon 1, 0 occurrences of taxon 2, 0 occurrences of taxon 3, 0 occurrences of taxon 4
    #     [0, 2, 0, 0],  # geohash 2 has 0 occurrences of taxon 1, 2 occurrences of taxon 2, 0 occurrences of taxon 3, 0 occurrences of taxon 4
    #     [0, 0, 3, 0],  # geohash 3 has 0 occurrences of taxon 1, 0 occurrences of taxon 2, 3 occurrences of taxon 3, 0 occurrences of taxon 4
    #     [0, 0, 0, 4],  # geohash 4 has 0 occurrences of taxon 1, 0 occurrences of taxon 2, 0 occurrences of taxon 3, 4 occurrences of taxon 4
    # ]
    matrix = np.zeros((len(ordered_seen_geohash), len(ordered_seen_taxon_id)))
    for i, geohash in enumerate(ordered_seen_geohash):
        for j, taxon_id in enumerate(ordered_seen_taxon_id):
            matrix[i, j] = read_rows_result.geohash_to_taxon_id_to_count[geohash].get(
                taxon_id, 0
            )

    # whitened = whiten(matrix)

    return ordered_seen_geohash, pdist(matrix, metric="braycurtis")


class Stats(NamedTuple):
    # taxon_id -> average per geohash
    averages: Dict[TaxonId, float]
    # taxon_id -> count
    counts: Counter[TaxonId]


def build_stats(
    read_rows_result: ReadRowsResult, geohash_filter: Optional[List[Geohash]] = None
) -> Stats:
    # taxon_id -> taxon average
    averages: DefaultDict[TaxonId, float] = defaultdict(float)
    # taxon_id -> taxon count
    counts: Counter[TaxonId] = Counter()

    # Calculate total counts for each taxon_id
    for geohash, taxon_counts in read_rows_result.geohash_to_taxon_id_to_count.items():
        if geohash_filter and geohash not in geohash_filter:
            continue
        for taxon_id, count in taxon_counts.items():
            counts[taxon_id] += count

    # Calculate averages for each taxon_id
    for taxon_id in counts.keys():
        averages[taxon_id] = counts[taxon_id] / sum(counts.values())

    return Stats(averages=averages, counts=counts)


def print_cluster_stats(
    cluster: int,
    geohashes: List[Geohash],
    read_rows_result: ReadRowsResult,
    all_stats: Stats,
) -> None:
    stats = build_stats(read_rows_result, geohash_filter=geohashes)
    print("-" * 10)
    print(f"cluster {cluster} (count: {len(geohashes)})")
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
        read_rows_result = build_read_rows_result(input_file, args.geohash_precision)
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
    all_stats = build_stats(read_rows_result)
    # For each top count taxon, print the average per geohash
    print("all")
    for taxon_id, count in sorted(
        all_stats.counts.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(
            f"{read_rows_result.taxon_index[taxon_id]}: {all_stats.averages[taxon_id]}"
        )

    # Generate the linkage matrix
    Z = linkage(condensed_distance_matrix, "ward")
    # fig = plt.figure(figsize=(25, 10))
    # dn = dendrogram(Z, labels=ordered_seen_geohash)
    # plt.show()

    clusters = fcluster(Z, t=5, criterion="maxclust")
    logger.info(f"Number of clusters: {len(set(clusters))}")

    geohash_to_cluster = {
        geohash: int(cluster)
        for geohash, cluster in zip(ordered_seen_geohash, clusters)
    }
    cluster_to_geohashes = {
        int(cluster): [g for g, c in geohash_to_cluster.items() if c == cluster]
        for cluster in set(clusters)
    }

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            build_geojson_feature(geohashes, cluster)
            for cluster, geohashes in cluster_to_geohashes.items()
        ],
    }

    for cluster, geohashes in cluster_to_geohashes.items():
        print_cluster_stats(cluster, geohashes, read_rows_result, all_stats)

    with open(args.output_file, "w") as geojson_writer:
        json.dump(feature_collection, geojson_writer)
