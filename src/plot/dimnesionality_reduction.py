from typing import Literal

import dataframely as dy
import numpy as np
import seaborn as sns
import umap
from sklearn.manifold import TSNE

from src.dataframes.cluster_color import ClusterColorSchema, to_dict
from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.matrices.cluster_distance import ClusterDistanceMatrix


def create_dimensionality_reduction_plot(
    geocode_distance_matrix: ClusterDistanceMatrix,
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
    cluster_color_df: dy.DataFrame[ClusterColorSchema],
    method: Literal["umap", "tsne"] = "umap",
    n_neighbors: int = 3000,
):
    """
    Create a dimensionality reduction plot using either UMAP or t-SNE.

    Parameters:
    -----------
    geocode_distance_matrix : object
        Distance matrix with squareform method
    geocode_cluster_df : object
        DataFrame containing cluster information
    cluster_colors_df : object
        Object with color mapping for clusters
    method : str, optional
        Dimensionality reduction method to use ('umap' or 'tsne'), default 'umap'
    n_neighbors : int, optional
        Number of neighbors for UMAP, default 3000

    Returns:
    --------
    matplotlib.axes.Axes
        The plot axes
    """
    if method == "tsne":
        tsne = TSNE(
            n_components=2,
            random_state=42,
            metric="precomputed",
            init="random",
            perplexity=min(
                30, geocode_distance_matrix.squareform().shape[0] - 1
            ),  # HACK FOR SMALLER DATASETS
        )
        X_reduced: np.ndarray = tsne.fit_transform(geocode_distance_matrix.squareform())
    else:  # umap
        umap_reducer = umap.UMAP(
            n_components=2,
            metric="precomputed",
            random_state=42,
            n_neighbors=n_neighbors,
            # min_dist=1,
            # init="random",
        )
        X_reduced = umap_reducer.fit_transform(geocode_distance_matrix.squareform())  # type: ignore

    return sns.scatterplot(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        hue=geocode_cluster_df["cluster"],
        palette=to_dict(cluster_color_df),
        alpha=1,
    )
