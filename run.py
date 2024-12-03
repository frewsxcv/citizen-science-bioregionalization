import csv
import itertools
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
from typing import Dict, Generator, List, NamedTuple, Set, Tuple
import argparse

COLORS = [
    "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
    for _ in range(1000)
]

logger = logging.getLogger(__name__)


class Row(NamedTuple):
    lat: float
    lon: float
    taxon_id: int

    def geohash(self, precision: int) -> str:
        return pygeohash.encode(self.lat, self.lon, precision=precision)


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
            taxon_id = read_int(row["taxonID"])
            if not taxon_id:
                logger.error(f"Invalid taxon ID: {row['taxonID']}")
                continue
            yield Row(lat, lon, taxon_id)


class Point(NamedTuple):
    lat: float
    lon: float


class Bbox(NamedTuple):
    sw: Point
    ne: Point


def geohash_to_bbox(geohash) -> Bbox:
    lat, lon, lat_err, lon_err = pygeohash.decode_exactly(geohash)
    return Bbox(
        sw=Point(lat=lat - lat_err, lon=lon - lon_err),
        ne=Point(lat=lat + lat_err, lon=lon + lon_err),
    )


def build_geojson_feature(geohash: str, cluster: int) -> Dict:
    bbox = geohash_to_bbox(geohash)
    coords = [
        [bbox.sw.lon, bbox.sw.lat],
        [bbox.ne.lon, bbox.sw.lat],
        [bbox.ne.lon, bbox.ne.lat],
        [bbox.sw.lon, bbox.ne.lat],
        [bbox.sw.lon, bbox.sw.lat],
    ]
    return {
        "type": "Feature",
        "properties": {
            "label": geohash,
            "fill": COLORS[int(cluster)],
            "stroke-width": 0,
            "cluster": int(cluster),
        },
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def build_condensed_distance_matrix(
    input_file: str, geohash_precision: int
) -> Tuple[List[str], np.ndarray]:
    geohash_to_taxon_id_to_count: Dict[str, Dict[int, int]] = {}
    seen_taxon_id: Set[int] = set()

    logger.info("Reading rows")

    for row in read_rows(input_file):
        geohash = row.geohash(geohash_precision)
        geohash_to_taxon_id_to_count[geohash] = geohash_to_taxon_id_to_count.get(
            geohash, {}
        )
        geohash_to_taxon_id_to_count[geohash][row.taxon_id] = (
            geohash_to_taxon_id_to_count[geohash].get(row.taxon_id, 0) + 1
        )
        seen_taxon_id.add(row.taxon_id)

    ordered_seen_taxon_id = sorted(seen_taxon_id)
    ordered_seen_geohash = sorted(geohash_to_taxon_id_to_count.keys())

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
            matrix[i, j] = geohash_to_taxon_id_to_count[geohash].get(taxon_id, 0)

    whitened = whiten(matrix)

    return ordered_seen_geohash, pdist(whitened, metric="braycurtis")


if __name__ == "__main__":
    args = parse_arguments()
    input_file = args.input_file
    logging.basicConfig(filename=args.log_file, encoding="utf-8", level=logging.DEBUG)

    if os.path.exists("condensed_distance_matrix.pickle"):
        logger.info("Loading condensed distance matrix")
        with open("condensed_distance_matrix.pickle", "rb") as pickle_reader:
            ordered_seen_geohash, condensed_distance_matrix = pickle.load(pickle_reader)
    else:
        ordered_seen_geohash, condensed_distance_matrix = (
            build_condensed_distance_matrix(input_file, args.geohash_precision)
        )
        logger.info("Saving condensed distance matrix")
        with open("condensed_distance_matrix.pickle", "wb") as pickle_writer:
            pickle.dump(
                (ordered_seen_geohash, condensed_distance_matrix), pickle_writer
            )

    # Generate the linkage matrix
    Z = linkage(condensed_distance_matrix, "ward")
    # fig = plt.figure(figsize=(25, 10))
    # dn = dendrogram(Z, labels=ordered_seen_geohash)
    # plt.show()

    clusters = fcluster(Z, t=3, criterion="distance")
    logger.info(f"Number of clusters: {len(set(clusters))}")

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            build_geojson_feature(geohash, cluster)
            for geohash, cluster in zip(ordered_seen_geohash, clusters)
        ],
    }
    with open(args.output_file, "w") as geojson_writer:
        json.dump(feature_collection, geojson_writer)
