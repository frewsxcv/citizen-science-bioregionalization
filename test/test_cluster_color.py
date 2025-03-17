import unittest
import polars as pl
import networkx as nx
import shapely
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.cluster_neighbors import ClusterNeighborsDataFrame
from src.dataframes.cluster_boundary import ClusterBoundaryDataFrame
from src.geojson import find_ocean_clusters


class TestClusterColorDataFrame(unittest.TestCase):
    def test_ocean_coloring(self):
        """Test that ocean clusters get blue colors and land clusters get non-blue colors"""
        
        # Create a simple cluster neighbors dataframe with correct schema
        neighbors_df = pl.DataFrame({
            "cluster": [1, 2, 3, 4],
            "direct_neighbors": [[2], [1, 3], [2, 4], [3]],
            "direct_and_indirect_neighbors": [[2, 3], [1, 3, 4], [1, 2, 4], [2, 3]]
        }, schema={
            "cluster": pl.UInt32(),
            "direct_neighbors": pl.List(pl.UInt32),
            "direct_and_indirect_neighbors": pl.List(pl.UInt32)
        })
        cluster_neighbors = ClusterNeighborsDataFrame(neighbors_df)
        
        # Create mock cluster boundaries
        # Clusters 1 and 3 will be within the ocean area (-5,-5 to 5,5)
        # Clusters 2 and 4 will be outside the ocean area (15,15 to 25,25)
        ocean_polygon = shapely.Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        land_polygon = shapely.Polygon([(15, 15), (25, 15), (25, 25), (15, 25)])
        
        ocean_wkb = shapely.to_wkb(ocean_polygon)
        land_wkb = shapely.to_wkb(land_polygon)
        
        boundaries_df = pl.DataFrame({
            "cluster": [1, 2, 3, 4],
            "boundary": [ocean_wkb, land_wkb, ocean_wkb, land_wkb]
        }).with_columns([
            pl.col("cluster").cast(pl.UInt32())
        ])
        cluster_boundaries = ClusterBoundaryDataFrame(boundaries_df)
        
        # Generate colors
        color_df = ClusterColorDataFrame.build(
            cluster_neighbors, cluster_boundaries
        )
        
        # Get the colors
        colors_dict = color_df.to_dict()
        
        # Check that clusters 1 and 3 have blue-ish colors
        for cluster in [1, 3]:
            color = colors_dict[cluster]
            # Convert hex to RGB and check if blue is dominant component
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            self.assertTrue(b > r and b > g, f"Cluster {cluster} should have blue-ish color, got {color}")
        
        # Check that clusters 2 and 4 have non-blue colors
        for cluster in [2, 4]:
            color = colors_dict[cluster]
            # Convert hex to RGB and check if blue is NOT dominant component
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            self.assertFalse(b > r and b > g, f"Cluster {cluster} should not have blue-ish color, got {color}")
            
    def test_color_separation(self):
        """Test that two adjacent clusters never have the same color"""
        # Create a cluster neighbors dataframe with a simple graph structure and correct schema
        neighbors_df = pl.DataFrame({
            "cluster": [1, 2, 3, 4],
            "direct_neighbors": [[2, 3], [1, 3], [1, 2, 4], [3]],
            "direct_and_indirect_neighbors": [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]]
        }, schema={
            "cluster": pl.UInt32(),
            "direct_neighbors": pl.List(pl.UInt32),
            "direct_and_indirect_neighbors": pl.List(pl.UInt32)
        })
        cluster_neighbors = ClusterNeighborsDataFrame(neighbors_df)
        
        # Create mock cluster boundaries
        # Same as above test
        ocean_polygon = shapely.Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        land_polygon = shapely.Polygon([(15, 15), (25, 15), (25, 25), (15, 25)])
        
        ocean_wkb = shapely.to_wkb(ocean_polygon)
        land_wkb = shapely.to_wkb(land_polygon)
        
        boundaries_df = pl.DataFrame({
            "cluster": [1, 2, 3, 4],
            "boundary": [ocean_wkb, land_wkb, ocean_wkb, land_wkb]
        }).with_columns([
            pl.col("cluster").cast(pl.UInt32())
        ])
        cluster_boundaries = ClusterBoundaryDataFrame(boundaries_df)
        
        # Generate colors
        color_df = ClusterColorDataFrame.build(
            cluster_neighbors, cluster_boundaries
        )
        
        # Get the colors
        colors_dict = color_df.to_dict()
        
        # Verify that adjacent clusters have different colors
        for cluster, neighbors in zip(neighbors_df["cluster"], neighbors_df["direct_neighbors"]):
            for neighbor in neighbors:
                self.assertNotEqual(
                    colors_dict[cluster], 
                    colors_dict[neighbor],
                    f"Adjacent clusters {cluster} and {neighbor} have the same color {colors_dict[cluster]}"
                )
