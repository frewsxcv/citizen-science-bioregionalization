import logging
from typing import Dict, List, Literal, Optional

import dataframely as dy
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import polars as pl
import seaborn as sns
import umap  # type: ignore
from sklearn.manifold import MDS  # type: ignore

from src.colors import darken_hex_color
from src.dataframes.cluster_neighbors import ClusterNeighborsSchema, to_graph
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema
from src.matrices.cluster_distance import ClusterDistanceMatrix
from src.types import ClusterId

logger = logging.getLogger(__name__)


class ClusterColorSchema(dy.Schema):
    cluster = dy.UInt32(nullable=False)
    color = dy.String(nullable=False)
    darkened_color = dy.String(nullable=False)


def build_cluster_color_df(
    cluster_neighbors_lf: dy.LazyFrame[ClusterNeighborsSchema],
    cluster_taxa_statistics_df: Optional[
        dy.DataFrame[ClusterTaxaStatisticsSchema]
    ] = None,
    color_method: Literal["geographic", "taxonomic"] = "geographic",
) -> dy.DataFrame[ClusterColorSchema]:
    """
    Build a ClusterColorSchema DataFrame using either geographic neighbor-based coloring
    or taxonomic similarity-based coloring.

    Args:
        cluster_neighbors_lf: Lazyframe of cluster neighbors
        cluster_taxa_statistics_df: Dataframe of cluster taxa statistics (required for taxonomic coloring)
        color_method: Method to use for coloring clusters ("geographic" or "taxonomic")

    Returns:
        A ClusterColorSchema DataFrame with colors assigned to clusters
    """
    logger.info(f"build_cluster_color_df: Starting with color_method={color_method}")

    if color_method == "geographic":
        df = _build_geographic(cluster_neighbors_lf)
    elif color_method == "taxonomic":
        assert cluster_taxa_statistics_df is not None, (
            "cluster_taxa_statistics_df is required for taxonomic coloring"
        )
        df = _build_taxonomic(cluster_taxa_statistics_df)
    else:
        raise ValueError(f"Invalid color_method: {color_method}")
    return ClusterColorSchema.validate(df)


def get_color_for_cluster(
    cluster_color_df: dy.DataFrame[ClusterColorSchema], cluster: ClusterId
) -> str:
    return cluster_color_df.filter(pl.col("cluster") == cluster)["color"].to_list()[0]


def to_dict(
    cluster_color_df: dy.DataFrame[ClusterColorSchema],
) -> Dict[ClusterId, str]:
    return {
        x: get_color_for_cluster(cluster_color_df, x)
        for x in cluster_color_df["cluster"]
    }


def _build_geographic(
    cluster_neighbors_lf: dy.LazyFrame[ClusterNeighborsSchema],
) -> pl.DataFrame:
    """
    Creates a coloring where neighboring clusters have different colors.
    """
    G = to_graph(cluster_neighbors_lf)

    # Use NetworkX to color the entire graph - this ensures adjacent nodes have different colors
    color_indices = nx.coloring.greedy_color(G)

    # Determine how many unique colors needed
    num_colors = len(set(color_indices.values()))

    # Generate color palette with the exact size needed
    palette = sns.color_palette("YlOrRd", num_colors).as_hex()

    # Create mapping from color indices to colors
    palette_map = dict(zip(sorted(set(color_indices.values())), palette))

    # Map color indices to actual colors
    rows = []
    for cluster, color_index in color_indices.items():
        color = palette_map[color_index]

        rows.append(
            {
                "cluster": cluster,
                "color": color,
                "darkened_color": darken_hex_color(color),
            }
        )

    return pl.DataFrame(rows).with_columns(pl.col("cluster").cast(pl.UInt32))


def _build_taxonomic(
    cluster_taxa_statistics_df: dy.DataFrame[ClusterTaxaStatisticsSchema],
) -> pl.DataFrame:
    """
    Creates a coloring where clusters with similar taxonomic composition
    have similar colors, using UMAP for dimensionality reduction.

    Requires at least 10 clusters to work properly with the UMAP algorithm.
    """
    # Build the distance matrix based on taxonomic composition
    distance_matrix = ClusterDistanceMatrix.build(cluster_taxa_statistics_df)

    clusters = distance_matrix.cluster_ids()

    # Get the square-form distance matrix for dimensionality reduction
    square_matrix = distance_matrix.squareform()

    # Assert that we have enough clusters for UMAP
    # UMAP has known issues with very small datasets when using precomputed metrics
    assert len(clusters) >= 10, (
        f"UMAP requires at least 10 clusters, got {len(clusters)}"
    )

    # Set appropriate parameters for UMAP
    n_components = 3  # Always use 3 dimensions for color mapping

    # Assert that we have enough samples for the chosen number of components
    assert len(clusters) > n_components + 1, (
        f"Need at least {n_components + 2} clusters for {n_components}D UMAP, got {len(clusters)}"
    )

    # Set n_neighbors to be less than number of clusters
    n_neighbors = min(len(clusters) - 1, 5)

    # Apply UMAP for dimensionality reduction to a color space
    reducer = umap.UMAP(
        n_components=n_components,
        metric="precomputed",
        min_dist=0.1,  # Smaller min_dist for better separation
        n_neighbors=n_neighbors,
        random_state=42,  # For reproducibility
    )

    positions: np.ndarray = reducer.fit_transform(square_matrix)  # type: ignore

    # Normalize positions to [0,1] range for RGB color mapping
    positions_normalized = np.zeros_like(positions)
    for i in range(positions.shape[1]):
        col_min = positions[:, i].min()
        col_max = positions[:, i].max()
        if col_max > col_min:
            positions_normalized[:, i] = (positions[:, i] - col_min) / (
                col_max - col_min
            )
        # If all values are the same, leave them as zeros

    # Create colors based on taxonomic similarity
    rows = []
    for i, cluster in enumerate(clusters):
        # Use a full spectrum of hues for all clusters
        h = positions_normalized[i, 0]  # Full hue range (0-1)
        s = 0.6 + (positions_normalized[i, 1] * 0.4)  # Saturation (0.6-1.0)
        v = 0.7 + (positions_normalized[i, 2] * 0.3)  # Value/brightness (0.7-1.0)
        rgb = mcolors.hsv_to_rgb([h, s, v])

        # Convert RGB to hex
        hex_color = mcolors.rgb2hex((rgb[0], rgb[1], rgb[2]))
        rows.append(
            {
                "cluster": cluster,
                "color": hex_color,
                "darkened_color": darken_hex_color(hex_color),
            }
        )

    return pl.DataFrame(rows).with_columns(pl.col("cluster").cast(pl.UInt32))
