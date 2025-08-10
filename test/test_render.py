import unittest
import os
import io
import json
import geojson  # type: ignore
import polars as pl
import polars_st  # Import for st attribute
from src.render import (
    plot_clusters,
    plot_single_cluster,
    plot_entire_region,
    features_to_polars_df,
    darken_hex_color,
    darken_hex_colors_polars,
)


class TestRender(unittest.TestCase):
    """Test cases for the render module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple GeoJSON feature collection for testing
        self.feature_collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-122.1, 47.5],
                                [-122.0, 47.5],
                                [-122.0, 47.6],
                                [-122.1, 47.6],
                                [-122.1, 47.5],
                            ]
                        ],
                    },
                    "properties": {
                        "cluster": 1,
                        "fillColor": "#ff0000",
                        "color": "#800000",
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-122.3, 47.7],
                                [-122.2, 47.7],
                                [-122.2, 47.8],
                                [-122.3, 47.8],
                                [-122.3, 47.7],
                            ]
                        ],
                    },
                    "properties": {
                        "cluster": 2,
                        "fillColor": "#00ff00",
                        "color": "#008000",
                    },
                },
            ],
        }

    def test_features_to_polars_df(self):
        """Test if features_to_polars_df correctly converts features to a DataFrame."""
        df = features_to_polars_df(self.feature_collection["features"])
        self.assertIsInstance(df, pl.DataFrame)
        self.assertIn("geometry", df.columns)
        self.assertIn("cluster", df.columns)
        self.assertIn("fillColor", df.columns)
        self.assertEqual(len(df), 2)

    def test_darken_hex_color(self):
        """Test if darken_hex_color correctly darkens a hex color."""
        # Test with a regular hex color
        self.assertEqual(darken_hex_color("#ff0000", 0.5), "#7f0000")
        # Test with a shorthand hex color
        self.assertEqual(darken_hex_color("#f00", 0.5), "#7f0000")
        # Test with different factors
        self.assertEqual(darken_hex_color("#ffffff", 0.0), "#000000")
        self.assertEqual(darken_hex_color("#ffffff", 1.0), "#ffffff")

    def test_plot_clusters(self):
        """Test if plot_clusters function runs without errors."""
        buffer = io.BytesIO()
        plot_clusters(self.feature_collection, file_obj=buffer)
        buffer.seek(0)
        self.assertGreater(len(buffer.read()), 0)

    def test_plot_single_cluster(self):
        """Test if plot_single_cluster function runs without errors."""
        buffer = io.BytesIO()
        plot_single_cluster(self.feature_collection, cluster_id=1, file_obj=buffer)
        buffer.seek(0)
        self.assertGreater(len(buffer.read()), 0)

    def test_darken_hex_colors_polars(self):
        """Test if darken_hex_colors_polars correctly darkens all colors in a Series."""
        colors = pl.Series(["#ff0000", "#00ff00", "#0000ff"])
        darkened = darken_hex_colors_polars(colors)
        self.assertEqual(darkened[0], "#7f0000")
        self.assertEqual(darkened[1], "#007f00")
        self.assertEqual(darkened[2], "#00007f")

    def test_plot_entire_region(self):
        """Test if plot_entire_region function runs without errors."""
        buffer = io.BytesIO()
        plot_entire_region(self.feature_collection, file_obj=buffer)
        buffer.seek(0)
        self.assertGreater(len(buffer.read()), 0)


if __name__ == "__main__":
    unittest.main()
