import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from inaturalist_ecoregions.dataframes.geocode_cluster import GeocodeClusterDataFrame
from inaturalist_ecoregions.matrices.geocode_distance import GeocodeDistanceMatrix
from inaturalist_ecoregions.dataframes.geocode_silhouette_score import GeocodeSilhouetteScoreDataFrame
from inaturalist_ecoregions.dataframes.cluster_color import ClusterColorDataFrame

def plot_silhouette_scores(
    geocode_cluster_dataframe: GeocodeClusterDataFrame,
    geocode_distance_matrix: GeocodeDistanceMatrix,
    geocode_silhouette_score_dataframe: GeocodeSilhouetteScoreDataFrame,
    cluster_colors_dataframe: ClusterColorDataFrame,
) -> plt.Figure:
    """
    Create a silhouette plot for clustering results.
    
    Args:
        geocode_cluster_dataframe: DataFrame containing cluster assignments
        geocode_distance_matrix: Matrix containing distances between geocodes
        geocode_silhouette_score_dataframe: DataFrame containing silhouette scores
        cluster_colors_dataframe: DataFrame containing color mappings for clusters
        
    Returns:
        matplotlib.Figure: The generated silhouette plot
    """
    n_clusters = len(geocode_cluster_dataframe.df["cluster"].unique())
    n_geocodes = len(geocode_distance_matrix.squareform())

    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots()
    fig.set_size_inches(18, 7)
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim(0, n_geocodes + (n_clusters + 1) * 10)

    y_lower = 10
    for i, cluster in enumerate(geocode_cluster_dataframe.df["cluster"].unique()):
        ith_cluster_silhouette_values = (
            geocode_silhouette_score_dataframe.df.join(
                geocode_cluster_dataframe.df, on="geocode"
            )
            .filter(pl.col("cluster") == cluster)
            .sort("silhouette_score", descending=True)
        )["silhouette_score"]

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cluster_colors_dataframe.get_color_for_cluster(cluster)
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