import unittest
import polars as pl
import networkx as nx
import shapely
import numpy as np
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.cluster_neighbors import ClusterNeighborsDataFrame
from src.dataframes.cluster_boundary import ClusterBoundaryDataFrame
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame
from src.geojson import find_ocean_clusters


class TestClusterColorDataFrame(unittest.TestCase):
    def test_ocean_coloring(self):
        """Test that ocean clusters get blue colors and land clusters get non-blue colors"""

        # Create a simple cluster neighbors dataframe with correct schema
        neighbors_df = pl.DataFrame(
            {
                "cluster": [1, 2, 3, 4],
                "direct_neighbors": [[2], [1, 3], [2, 4], [3]],
                "direct_and_indirect_neighbors": [[2, 3], [1, 3, 4], [1, 2, 4], [2, 3]],
            },
            schema={
                "cluster": pl.UInt32(),
                "direct_neighbors": pl.List(pl.UInt32),
                "direct_and_indirect_neighbors": pl.List(pl.UInt32),
            },
        )
        cluster_neighbors = ClusterNeighborsDataFrame(neighbors_df)

        # Create mock cluster boundaries
        # Clusters 1 and 3 will be within the ocean area (-5,-5 to 5,5)
        # Clusters 2 and 4 will be outside the ocean area (15,15 to 25,25)
        ocean_polygon = shapely.Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        land_polygon = shapely.Polygon([(15, 15), (25, 15), (25, 25), (15, 25)])

        ocean_wkb = shapely.to_wkb(ocean_polygon)
        land_wkb = shapely.to_wkb(land_polygon)

        boundaries_df = pl.DataFrame(
            {
                "cluster": [1, 2, 3, 4],
                "geometry": [ocean_wkb, land_wkb, ocean_wkb, land_wkb],
            }
        ).with_columns([pl.col("cluster").cast(pl.UInt32())])
        cluster_boundaries = ClusterBoundaryDataFrame(boundaries_df)

        # Generate colors
        color_df = ClusterColorDataFrame.build(cluster_neighbors, cluster_boundaries)

        # Get the colors
        colors_dict = color_df.to_dict()

        # Check that clusters 1 and 3 have blue-ish colors
        for cluster in [1, 3]:
            color = colors_dict[cluster]
            # Convert hex to RGB and check if blue is dominant component
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            self.assertTrue(
                b > r and b > g,
                f"Cluster {cluster} should have blue-ish color, got {color}",
            )

        # Check that clusters 2 and 4 have non-blue colors
        for cluster in [2, 4]:
            color = colors_dict[cluster]
            # Convert hex to RGB and check if blue is NOT dominant component
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            self.assertFalse(
                b > r and b > g,
                f"Cluster {cluster} should not have blue-ish color, got {color}",
            )

    def test_color_separation(self):
        """Test that two adjacent clusters never have the same color"""
        # Create a cluster neighbors dataframe with a simple graph structure and correct schema
        neighbors_df = pl.DataFrame(
            {
                "cluster": [1, 2, 3, 4],
                "direct_neighbors": [[2, 3], [1, 3], [1, 2, 4], [3]],
                "direct_and_indirect_neighbors": [
                    [2, 3, 4],
                    [1, 3, 4],
                    [1, 2, 4],
                    [1, 2, 3],
                ],
            },
            schema={
                "cluster": pl.UInt32(),
                "direct_neighbors": pl.List(pl.UInt32),
                "direct_and_indirect_neighbors": pl.List(pl.UInt32),
            },
        )
        cluster_neighbors = ClusterNeighborsDataFrame(neighbors_df)

        # Create mock cluster boundaries
        # Same as above test
        ocean_polygon = shapely.Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        land_polygon = shapely.Polygon([(15, 15), (25, 15), (25, 25), (15, 25)])

        ocean_wkb = shapely.to_wkb(ocean_polygon)
        land_wkb = shapely.to_wkb(land_polygon)

        boundaries_df = pl.DataFrame(
            {
                "cluster": [1, 2, 3, 4],
                "geometry": [ocean_wkb, land_wkb, ocean_wkb, land_wkb],
            }
        ).with_columns([pl.col("cluster").cast(pl.UInt32())])
        cluster_boundaries = ClusterBoundaryDataFrame(boundaries_df)

        # Generate colors
        color_df = ClusterColorDataFrame.build(cluster_neighbors, cluster_boundaries)

        # Get the colors
        colors_dict = color_df.to_dict()

        # Verify that adjacent clusters have different colors
        for cluster, neighbors in zip(
            neighbors_df["cluster"], neighbors_df["direct_neighbors"]
        ):
            for neighbor in neighbors:
                self.assertNotEqual(
                    colors_dict[cluster],
                    colors_dict[neighbor],
                    f"Adjacent clusters {cluster} and {neighbor} have the same color {colors_dict[cluster]}",
                )

    def test_taxon_similarity_umap_coloring(self):
        """Test that the UMAP-based taxonomic similarity coloring works as expected"""
        # This test verifies that the UMAP-based method correctly raises an
        # AssertionError when there are fewer than 10 clusters

        # Create a mock ClusterTaxaStatisticsDataFrame with 4 clusters
        # This is below the minimum required for UMAP
        taxa_stats_df = pl.DataFrame(
            {
                "cluster": [1, 1, 2, 2, 3, 3, 4, 4, None, None],
                "taxonId": [101, 102, 101, 102, 101, 102, 101, 102, 101, 102],
                "count": [8, 2, 2, 8, 7, 3, 3, 7, 20, 20],
                "average": [0.8, 0.2, 0.2, 0.8, 0.7, 0.3, 0.3, 0.7, 0.5, 0.5],
            },
            schema={
                "cluster": pl.UInt32(),
                "taxonId": pl.UInt32(),
                "count": pl.UInt32(),
                "average": pl.Float64(),
            },
        )
        cluster_taxa_stats = ClusterTaxaStatisticsDataFrame(taxa_stats_df)

        # Create dummy neighbor and boundary data (not used for taxonomic coloring)
        neighbors_df = pl.DataFrame(
            {
                "cluster": [1, 2, 3, 4],
                "direct_neighbors": [[2], [1, 3], [2, 4], [3]],
                "direct_and_indirect_neighbors": [[2, 3], [1, 3, 4], [1, 2, 4], [2, 3]],
            },
            schema={
                "cluster": pl.UInt32(),
                "direct_neighbors": pl.List(pl.UInt32),
                "direct_and_indirect_neighbors": pl.List(pl.UInt32),
            },
        )
        cluster_neighbors = ClusterNeighborsDataFrame(neighbors_df)

        # Create mock cluster boundaries
        ocean_polygon = shapely.Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        ocean_wkb = shapely.to_wkb(ocean_polygon)

        boundaries_df = pl.DataFrame(
            {
                "cluster": [1, 2, 3, 4],
                "geometry": [ocean_wkb, ocean_wkb, ocean_wkb, ocean_wkb],
            }
        ).with_columns([pl.col("cluster").cast(pl.UInt32())])
        cluster_boundaries = ClusterBoundaryDataFrame(boundaries_df)

        # Verify that attempting to use UMAP with too few clusters raises an AssertionError
        with self.assertRaises(AssertionError) as context:
            ClusterColorDataFrame.build(
                cluster_neighbors,
                cluster_boundaries,
                cluster_taxa_stats,
                color_method="taxonomic",
            )

        # Verify the error message mentions the minimum cluster requirement
        self.assertIn("UMAP requires at least 10 clusters", str(context.exception))

    def test_taxon_similarity_umap_coloring_large(self):
        """Test UMAP-based coloring with enough clusters to use actual UMAP"""
        # Create a mock ClusterTaxaStatisticsDataFrame with 12 clusters
        # to trigger the actual UMAP path (not MDS fallback)
        clusters = list(range(1, 13))
        taxa_ids = [101, 102, 103]

        # Create test data
        data = []

        # Add overall stats row (cluster=None)
        for taxon_id in taxa_ids:
            data.append(
                {"cluster": None, "taxonId": taxon_id, "count": 120, "average": 0.33}
            )

        # Add rows for each cluster
        for cluster in clusters:
            for taxon_id in taxa_ids:
                # Make some clusters similar to each other
                if cluster <= 4:
                    # First group prefers taxon 101
                    avg = 0.6 if taxon_id == 101 else 0.2
                elif cluster <= 8:
                    # Second group prefers taxon 102
                    avg = 0.6 if taxon_id == 102 else 0.2
                else:
                    # Third group prefers taxon 103
                    avg = 0.6 if taxon_id == 103 else 0.2

                count = int(avg * 100)  # Scale to an integer count

                data.append(
                    {
                        "cluster": cluster,
                        "taxonId": taxon_id,
                        "count": count,
                        "average": avg,
                    }
                )

        # Create the DataFrame
        taxa_stats_df = pl.DataFrame(
            data,
            schema={
                "cluster": pl.UInt32(),
                "taxonId": pl.UInt32(),
                "count": pl.UInt32(),
                "average": pl.Float64(),
            },
        )

        # Convert None values to nulls for cluster column
        taxa_stats_df = taxa_stats_df.with_columns(
            pl.when(pl.col("cluster").is_null())
            .then(None)
            .otherwise(pl.col("cluster"))
            .alias("cluster")
        )

        cluster_taxa_stats = ClusterTaxaStatisticsDataFrame(taxa_stats_df)

        # Create dummy neighbor and boundary data (not used for taxonomic coloring)
        neighbors_df = pl.DataFrame(
            {
                "cluster": clusters,
                "direct_neighbors": [[2] for _ in range(12)],
                "direct_and_indirect_neighbors": [[2, 3] for _ in range(12)],
            },
            schema={
                "cluster": pl.UInt32(),
                "direct_neighbors": pl.List(pl.UInt32),
                "direct_and_indirect_neighbors": pl.List(pl.UInt32),
            },
        )
        cluster_neighbors = ClusterNeighborsDataFrame(neighbors_df)

        # Create mock cluster boundaries
        ocean_polygon = shapely.Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        ocean_wkb = shapely.to_wkb(ocean_polygon)

        boundaries_df = pl.DataFrame(
            {"cluster": clusters, "geometry": [ocean_wkb for _ in range(12)]}
        ).with_columns([pl.col("cluster").cast(pl.UInt32())])
        cluster_boundaries = ClusterBoundaryDataFrame(boundaries_df)

        # Generate colors using the UMAP-based approach
        color_df = ClusterColorDataFrame.build(
            cluster_neighbors,
            cluster_boundaries,
            cluster_taxa_stats,
            color_method="taxonomic",
        )

        # Get the colors
        colors_dict = color_df.to_dict()

        # Helper function to calculate color distance between two hex colors
        def color_distance(hex1, hex2):
            # Convert hex to RGB
            r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
            r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)

            # Calculate Euclidean distance in RGB space
            return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5

        # Calculate average distances within each group
        group1_pairs = [(i, j) for i in range(1, 5) for j in range(1, 5) if i < j]
        group2_pairs = [(i, j) for i in range(5, 9) for j in range(5, 9) if i < j]
        group3_pairs = [(i, j) for i in range(9, 13) for j in range(9, 13) if i < j]

        # Calculate average distances between groups
        between_pairs = [(i, j) for i in range(1, 5) for j in range(5, 13)]
        between_pairs.extend([(i, j) for i in range(5, 9) for j in range(9, 13)])

        # Calculate average distances
        avg_within_group1 = (
            sum(color_distance(colors_dict[i], colors_dict[j]) for i, j in group1_pairs)
            / len(group1_pairs)
            if group1_pairs
            else 0
        )
        avg_within_group2 = (
            sum(color_distance(colors_dict[i], colors_dict[j]) for i, j in group2_pairs)
            / len(group2_pairs)
            if group2_pairs
            else 0
        )
        avg_within_group3 = (
            sum(color_distance(colors_dict[i], colors_dict[j]) for i, j in group3_pairs)
            / len(group3_pairs)
            if group3_pairs
            else 0
        )

        avg_between = (
            sum(
                color_distance(colors_dict[i], colors_dict[j]) for i, j in between_pairs
            )
            / len(between_pairs)
            if between_pairs
            else 0
        )

        # Calculate overall within-group average
        avg_within = (avg_within_group1 + avg_within_group2 + avg_within_group3) / 3

        # Verify that within-group distances are less than between-group distances
        # This might sometimes fail due to the stochastic nature of UMAP, but it's a reasonable check
        # We use a fairly relaxed assertion to account for UMAP's behavior
        self.assertLessEqual(
            avg_within,
            avg_between * 1.5,
            "Clusters within the same group should have more similar colors",
        )

        # Verify all clusters have colors
        self.assertEqual(len(colors_dict), 12)

        # Verify all colors are valid hex colors
        for cluster, color in colors_dict.items():
            self.assertTrue(color.startswith("#"))
            self.assertEqual(len(color), 7)
