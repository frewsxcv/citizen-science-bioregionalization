import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from typing import List
import numpy as np
from src import geohash


def show(Z: np.ndarray, ordered_seen_geohash: List[geohash.Geohash]) -> None:
    plt.figure()
    dendrogram(
        Z,
        labels=ordered_seen_geohash,
        leaf_label_func=lambda id: (
            ordered_seen_geohash[id]
            if geohash.is_water(ordered_seen_geohash[id])
            else ""
        ),
    )
    plt.show()
