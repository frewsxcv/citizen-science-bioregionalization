import unittest

import polars as pl

import geojson
from src.colors import darken_hex_color, darken_hex_colors_polars
from src.render import features_to_polars_df


class TestRender(unittest.TestCase):
    """Test cases for the render module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple GeoJSON feature collection for testing
        self.feature_collection = geojson.FeatureCollection(
            [
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
        )

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

    def test_darken_hex_colors_polars(self):
        """Test if darken_hex_colors_polars correctly darkens all colors in a Series."""
        colors = pl.Series(["#ff0000", "#00ff00", "#0000ff"])
        darkened = darken_hex_colors_polars(colors)
        self.assertEqual(darkened[0], "#7f0000")
        self.assertEqual(darkened[1], "#007f00")
        self.assertEqual(darkened[2], "#00007f")


if __name__ == "__main__":
    unittest.main()
