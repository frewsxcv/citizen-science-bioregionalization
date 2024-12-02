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
from typing import Dict, Generator, NamedTuple, Set

GEOHASH_PRECISION = 4
NUM_CLUSTERS = 5
COLORS = [
    "#"+''.join([random.choice('0123456789ABCDEF')
    for _ in range(6)])
    for _ in range(1000)
]

logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

class Row(NamedTuple):
    lat: float
    lon: float
    taxon_id: int

def read_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return None

def read_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return None

def read_rows() -> Generator[Row, None, None]:
    with open('occurrence.txt', 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            lat, lon = read_float(row['decimalLatitude']), read_float(row['decimalLongitude'])
            if not (lat and lon):
                logger.error(f"Invalid latitude or longitude: {row['decimalLatitude']}, {row['decimalLongitude']}")
                continue
            taxon_id = read_int(row['taxonID'])
            if not taxon_id:
                logger.error(f"Invalid taxon ID: {row['taxonID']}")
                continue
            yield Row(lat, lon, taxon_id)

class Point(NamedTuple):
    lat: float
    lon: float

# bbox lower left, bbox upper right
def geohash_to_rect(geohash):
    lat, lon, lat_err, lon_err = pygeohash.decode_exactly(geohash)
    return Point(
        lat=lat - lat_err,
        lon=lon - lon_err
    ), Point(
        lat=lat + lat_err,
        lon=lon + lon_err
    )

def build_geojson_feature(geohash: str, cluster: int):
    sw_bbox, ne_bbox = geohash_to_rect(geohash)
    coords = [
        [sw_bbox.lon, sw_bbox.lat],
        [ne_bbox.lon, sw_bbox.lat],
        [ne_bbox.lon, ne_bbox.lat],
        [sw_bbox.lon, ne_bbox.lat],
        [sw_bbox.lon, sw_bbox.lat],
    ]
    return {
        "type": "Feature",
        "properties": {
            "label": geohash,
            "fill": COLORS[int(cluster)],
            "cluster": int(cluster),
        },
        "geometry": {"type": "Polygon", "coordinates": [coords]}
    }

# # Each row of the M by N array is an observation vector. The columns are the features seen during each observation. The features must be whitened first with the whiten function.
# matrix = np.zeros((len(ordered_seen_geohash), len(ordered_seen_taxon_id)))
# for i, geohash in enumerate(ordered_seen_geohash):
#     for j, taxon_id in enumerate(ordered_seen_taxon_id):
#         matrix[i, j] = geohash_to_taxon_id_to_count[geohash].get(taxon_id, 0)

# whitened = whiten(matrix)
# centroid, label = kmeans2(whitened, NUM_CLUSTERS)

# def build_geojson_featurecollection():
#     features = []
#     for i, geohash in enumerate(ordered_seen_geohash):
#         features.append(build_geojson_feature(geohash, label[i]))
#     return {
#         "type": "FeatureCollection",
#         "features": features
#     }

# print(
#     json.dumps(
#         build_geojson_featurecollection()
#     )
# )



def build_condensed_distance_matrix():
    geohash_to_taxon_id_to_count: Dict[str, Dict[int, int]] = {}
    seen_taxon_id: Set[int] = set()

    logger.info("Reading rows")

    for row in read_rows():
        geohash = pygeohash.encode(row.lat, row.lon, precision=GEOHASH_PRECISION)
        geohash_to_taxon_id_to_count[geohash] = geohash_to_taxon_id_to_count.get(geohash, {})
        geohash_to_taxon_id_to_count[geohash][row.taxon_id] = geohash_to_taxon_id_to_count[geohash].get(row.taxon_id, 0) + 1
        seen_taxon_id.add(row.taxon_id)

    ordered_seen_taxon_id = sorted(seen_taxon_id)
    ordered_seen_geohash = sorted(geohash_to_taxon_id_to_count.keys())

    logger.info("Building condensed distance matrix")

    matrix = np.zeros((len(ordered_seen_geohash), len(ordered_seen_taxon_id)))
    for i, geohash in enumerate(ordered_seen_geohash):
        for j, taxon_id in enumerate(ordered_seen_taxon_id):
            matrix[i, j] = geohash_to_taxon_id_to_count[geohash].get(taxon_id, 0)

    return ordered_seen_geohash, pdist(matrix, metric='braycurtis')

if os.path.exists('condensed_distance_matrix.pickle'):
    logger.info("Loading condensed distance matrix")
    with open('condensed_distance_matrix.pickle', 'rb') as f:
        ordered_seen_geohash, condensed_distance_matrix = pickle.load(f)
else:
    ordered_seen_geohash, condensed_distance_matrix = build_condensed_distance_matrix()
    logger.info("Saving condensed distance matrix")
    with open('condensed_distance_matrix.pickle', 'wb') as f:
        pickle.dump((ordered_seen_geohash, condensed_distance_matrix), f)

# Generate the linkage matrix
Z = linkage(condensed_distance_matrix, 'complete')
# fig = plt.figure(figsize=(25, 10))
# dn = dendrogram(Z, labels=ordered_seen_geohash)
# plt.show()

clusters = fcluster(Z, t=0.97, criterion='distance')

feature_collection = {
    "type": "FeatureCollection",
    "features": [
        build_geojson_feature(geohash, cluster)
        for geohash, cluster in zip(ordered_seen_geohash, clusters)
    ]
}
with open('clusters.geojson', 'w') as f:
    json.dump(feature_collection, f)

