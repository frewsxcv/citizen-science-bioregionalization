from typing import Dict, List, Literal, Optional, Self
import polars as pl
import dataframely as dy
from src.dataframes.cluster_neighbors import ClusterNeighborsDataFrame
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.cluster_boundary import ClusterBoundarySchema
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame
from src.matrices.cluster_distance import ClusterDistanceMatrix
from src.types import ClusterId
from src.data_container import DataContainer, assert_dataframe_schema
import seaborn as sns
import networkx as nx
import numpy as np
from sklearn.manifold import MDS  # type: ignore
import matplotlib.colors as mcolors
import umap  # type: ignore


class ClusterColorDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "cluster": pl.UInt32(),
        "color": pl.Utf8(),
        "darkened_color": pl.Utf8(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    def get_color_for_cluster(self, cluster: ClusterId) -> str:
        return self.df.filter(pl.col("cluster") == cluster)["color"].to_list()[0]

    @classmethod
    def build(
        cls,
        cluster_neighbors_dataframe: ClusterNeighborsDataFrame,
        cluster_boundary_dataframe: dy.DataFrame[ClusterBoundarySchema],
        cluster_taxa_statistics_dataframe: Optional[
            ClusterTaxaStatisticsDataFrame
        ] = None,
        color_method: Literal["geographic", "taxonomic"] = "geographic",
        ocean_threshold: float = 0.90,
    ) -> Self:
        """
        Build a ClusterColorDataFrame using either geographic neighbor-based coloring
        or taxonomic similarity-based coloring.

        Args:
            cluster_neighbors_dataframe: Dataframe of cluster neighbors
            cluster_boundary_dataframe: Dataframe of cluster boundaries
            cluster_taxa_statistics_dataframe: Dataframe of cluster taxa statistics (required for taxonomic coloring)
            color_method: Method to use for coloring clusters ("geographic" or "taxonomic")
            ocean_threshold: Threshold for determining ocean clusters (only used with geographic method)

        Returns:
            A ClusterColorDataFrame with colors assigned to clusters
        """
        if color_method == "geographic":
            return cls._build_geographic(
                cluster_neighbors_dataframe, cluster_boundary_dataframe, ocean_threshold
            )
        elif color_method == "taxonomic":
            assert (
                cluster_taxa_statistics_dataframe is not None
            ), "cluster_taxa_statistics_dataframe is required for taxonomic coloring"
            return cls._build_taxonomic(cluster_taxa_statistics_dataframe)
        else:
            raise ValueError(f"Invalid color_method: {color_method}")

    @classmethod
    def _build_geographic(
        cls,
        cluster_neighbors_dataframe: ClusterNeighborsDataFrame,
        cluster_boundary_dataframe: dy.DataFrame[ClusterBoundarySchema],
        ocean_threshold: float = 0.90,
    ) -> Self:
        """
        Creates a coloring where neighboring clusters have different colors,
        and ocean and land clusters have different color palettes.
        """
        # Import here to avoid circular imports
        from src.geojson import find_ocean_clusters

        G = cluster_neighbors_dataframe.graph()

        # Find ocean clusters
        ocean_clusters = set(
            find_ocean_clusters(cluster_boundary_dataframe, threshold=ocean_threshold)
        )

        # Find land clusters
        land_clusters = set(G.nodes()) - ocean_clusters

        # Use NetworkX to color the entire graph - this ensures adjacent nodes have different colors
        color_indices = nx.coloring.greedy_color(G)
        ocean_color_indices = {
            cluster: color_indices[cluster] for cluster in ocean_clusters
        }
        land_color_indices = {
            cluster: color_indices[cluster] for cluster in land_clusters
        }

        # Determine how many unique colors needed for each group
        num_ocean_colors = len(set(ocean_color_indices.values()))
        num_land_colors = len(set(land_color_indices.values()))

        # Generate color palettes with the exact sizes needed
        ocean_palette = sns.color_palette("Blues", num_ocean_colors).as_hex()
        land_palette = sns.color_palette("YlOrRd", num_land_colors).as_hex()

        # Create mapping from color indices to colors
        ocean_palette_map = dict(
            zip(sorted(set(ocean_color_indices.values())), ocean_palette)
        )
        land_palette_map = dict(
            zip(sorted(set(land_color_indices.values())), land_palette)
        )

        # Map color indices to actual colors
        rows = []
        for cluster, color_index in color_indices.items():
            if cluster in ocean_clusters:
                color = ocean_palette_map[color_index]
            else:
                color = land_palette_map[color_index]

            rows.append(
                {
                    "cluster": cluster,
                    "color": color,
                    "darkened_color": darken_hex_color(color),
                }
            )

        return cls(pl.DataFrame(rows, schema=cls.SCHEMA))

    @classmethod
    def _build_taxonomic(
        cls,
        cluster_taxa_statistics_dataframe: ClusterTaxaStatisticsDataFrame,
    ) -> Self:
        """
        Creates a coloring where clusters with similar taxonomic composition
        have similar colors, using UMAP for dimensionality reduction.

        Requires at least 10 clusters to work properly with the UMAP algorithm.
        """
        # Build the distance matrix based on taxonomic composition
        distance_matrix = ClusterDistanceMatrix.build(cluster_taxa_statistics_dataframe)

        clusters = distance_matrix.cluster_ids()

        # Get the square-form distance matrix for dimensionality reduction
        square_matrix = distance_matrix.squareform()

        # Assert that we have enough clusters for UMAP
        # UMAP has known issues with very small datasets when using precomputed metrics
        assert (
            len(clusters) >= 10
        ), f"UMAP requires at least 10 clusters, got {len(clusters)}"

        # Set appropriate parameters for UMAP
        n_components = 3  # Always use 3 dimensions for color mapping

        # Assert that we have enough samples for the chosen number of components
        assert (
            len(clusters) > n_components + 1
        ), f"Need at least {n_components + 2} clusters for {n_components}D UMAP, got {len(clusters)}"

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

        positions = reducer.fit_transform(square_matrix)

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

        return cls(pl.DataFrame(rows, schema=cls.SCHEMA))

    def to_dict(self) -> Dict[ClusterId, str]:
        return {x: self.get_color_for_cluster(x) for x in self.df["cluster"]}


def darken_hex_color(hex_color: str, factor: float = 0.5) -> str:
    """
    Darkens a hex color by multiplying RGB components by the given factor.

    Args:
        hex_color: A hex color string like '#ff0000' or '#f00'
        factor: A float between 0 and 1 (0 = black, 1 = original color)

    Returns:
        A darkened hex color string
    """
    # Remove the # if present
    hex_color = hex_color.lstrip("#")

    # Handle shorthand hex format (#rgb)
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Darken each component
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"
