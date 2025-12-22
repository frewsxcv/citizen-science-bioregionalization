import dataframely as dy
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.dataframes.cluster_color import ClusterColorSchema, get_color_for_cluster
from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.dataframes.geocode_silhouette_score import GeocodeSilhouetteScoreSchema
from src.matrices.geocode_distance import GeocodeDistanceMatrix


def plot_silhouette_scores(
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
    geocode_distance_matrix: GeocodeDistanceMatrix,
    geocode_silhouette_score_df: dy.DataFrame[GeocodeSilhouetteScoreSchema],
    cluster_colors_df: dy.DataFrame[ClusterColorSchema],
) -> plt.Figure:  # type: ignore
    """
    Create a silhouette plot for clustering results.

    Args:
        geocode_cluster_df: DataFrame containing cluster assignments
        geocode_distance_matrix: Matrix containing distances between geocodes
        geocode_silhouette_score_df: DataFrame containing silhouette scores
        cluster_colors_df: DataFrame containing color mappings for clusters

    Returns:
        matplotlib.Figure: The generated silhouette plot
    """
    n_clusters = len(geocode_cluster_df["cluster"].unique())
    n_geocodes = len(geocode_distance_matrix.squareform())

    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots()
    fig.set_size_inches(18, 7)

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim(0, n_geocodes + (n_clusters + 1) * 10)

    y_lower = 10
    for i, cluster in enumerate(geocode_cluster_df["cluster"].unique()):
        ith_cluster_silhouette_values = (
            geocode_silhouette_score_df.join(geocode_cluster_df, on="geocode")
            .filter(pl.col("cluster") == cluster)
            .sort("silhouette_score", descending=True)
        )["silhouette_score"]

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = get_color_for_cluster(cluster_colors_df, cluster)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
        )

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    return fig
