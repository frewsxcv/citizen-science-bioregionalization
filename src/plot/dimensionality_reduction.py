from typing import Literal

import altair as alt
import dataframely as dy
import numpy as np
import polars as pl
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
) -> alt.Chart:
    """
    Create a dimensionality reduction plot using either UMAP or t-SNE.

    Parameters:
    -----------
    geocode_distance_matrix : ClusterDistanceMatrix
        Distance matrix with squareform method
    geocode_cluster_df : dy.DataFrame[GeocodeClusterSchema]
        DataFrame containing cluster information
    cluster_color_df : dy.DataFrame[ClusterColorSchema]
        DataFrame with color mapping for clusters
    method : str, optional
        Dimensionality reduction method to use ('umap' or 'tsne'), default 'umap'
    n_neighbors : int, optional
        Number of neighbors for UMAP, default 3000

    Returns:
    --------
    alt.Chart
        The Altair chart object
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
        )
        X_reduced = umap_reducer.fit_transform(geocode_distance_matrix.squareform())  # type: ignore

    # Build a Polars DataFrame for Altair
    plot_df = pl.DataFrame(
        {
            "x": X_reduced[:, 0],
            "y": X_reduced[:, 1],
            "cluster": geocode_cluster_df["cluster"].cast(pl.Utf8),
        }
    )

    # Get color mapping from cluster_color_df
    color_map = to_dict(cluster_color_df)
    # Convert keys to strings to match the cluster column
    color_domain = [str(k) for k in sorted(color_map.keys())]
    color_range = [color_map[k] for k in sorted(color_map.keys())]

    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x=alt.X("x:Q", title=f"{method.upper()} 1", axis=alt.Axis(grid=True)),
            y=alt.Y("y:Q", title=f"{method.upper()} 2", axis=alt.Axis(grid=True)),
            color=alt.Color(
                "cluster:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title="Cluster"),
            ),
            tooltip=["cluster:N", "x:Q", "y:Q"],
        )
        .properties(
            width=500,
            height=400,
            title=f"{method.upper()} Dimensionality Reduction",
        )
        .interactive()
    )

    return chart
