from typing import Literal
from sklearn.manifold import TSNE
import umap
import seaborn as sns
from src.matrices.cluster_distance import ClusterDistanceMatrix
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.cluster_color import ClusterColorDataFrame


def create_dimensionality_reduction_plot(
    geocode_distance_matrix: ClusterDistanceMatrix,
    geocode_cluster_dataframe: GeocodeClusterDataFrame,
    cluster_color_dataframe: ClusterColorDataFrame,
    method: Literal["umap", "tsne"] = "umap",
    n_neighbors: int = 3000,
):
    """
    Create a dimensionality reduction plot using either UMAP or t-SNE.

    Parameters:
    -----------
    geocode_distance_matrix : object
        Distance matrix with squareform method
    geocode_cluster_dataframe : object
        DataFrame containing cluster information
    cluster_colors_dataframe : object
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
        X_reduced = tsne.fit_transform(geocode_distance_matrix.squareform())
    else:  # umap
        umap_reducer = umap.UMAP(
            n_components=2,
            metric="precomputed",
            random_state=42,
            n_neighbors=n_neighbors,
            # min_dist=1,
            # init="random",
        )
        X_reduced = umap_reducer.fit_transform(geocode_distance_matrix.squareform())

    return sns.scatterplot(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        hue=geocode_cluster_dataframe.df["cluster"],
        palette=cluster_color_dataframe.to_dict(),
        alpha=1,
    )
