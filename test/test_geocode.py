import os
import unittest

import dataframely as dy
import polars as pl
import polars_st as pl_st
from dataframely.exc import ValidationError

from src.dataframes.darwin_core import build_darwin_core_lf
from src.dataframes.geocode import (
    GeocodeNoEdgesSchema,
    GeocodeSchema,
    build_geocode_df,
    build_geocode_no_edges_lf,
    index_of_geocode,
)
from src.types import Bbox


class TestGeocodeSchema(unittest.TestCase):
    def test_geocode_df_schema(self):
        """Test that the GeocodeSchema validates its schema correctly"""
        # Create a simple df that matches the expected schema
        df = pl.DataFrame(
            {
                "geocode": pl.Series(
                    [0x8514355555555555, 0x8514355555555557], dtype=pl.UInt64
                ),
                "center": points_series(2),
                "boundary": polygon_series(2),
                "is_edge": pl.Series([True, True], dtype=pl.Boolean),
            }
        )

        # This should not raise an exception
        geocode_df = GeocodeSchema.validate(df)
        self.assertIsInstance(geocode_df, pl.DataFrame)

    def test_build_from_darwin_core_csv(self):
        """Test building a GeocodeDataFrame from a LazyFrame"""
        bounding_box = Bbox.from_coordinates(-90.0, 90.0, -180.0, 180.0)

        # Build the LazyFrame from Darwin Core archive path
        darwin_core_lf = build_darwin_core_lf(
            source_path=os.path.join("test", "sample-archive"),
            bounding_box=bounding_box,
            limit=10,
        )

        geocode_df = build_geocode_df(
            darwin_core_lf,
            geocode_precision=8,
            bounding_box=bounding_box,
        )

        # Validate the result
        self.assertIsInstance(geocode_df, pl.DataFrame)

        # Should have unique geocodes
        self.assertEqual(
            geocode_df["geocode"].n_unique(),
            geocode_df.shape[0],
            "Geocodes should be unique",
        )

        # Check that we have the expected columns
        self.assertIn("geocode", geocode_df.columns)
        self.assertIn("center", geocode_df.columns)
        self.assertIn("boundary", geocode_df.columns)
        self.assertIn("is_edge", geocode_df.columns)

    def test_geocode_no_edges_schema(self):
        """Test that GeocodeNoEdgesSchema validates edge constraint"""
        # Create a df with no edge hexagons (should pass)
        df_no_edges = pl.DataFrame(
            {
                "geocode": pl.Series([1, 2], dtype=pl.UInt64),
                "center": points_series(2),
                "boundary": polygon_series(2),
                "is_edge": pl.Series([False, False], dtype=pl.Boolean),
            }
        )

        # This should not raise an exception
        geocode_df = GeocodeNoEdgesSchema.validate(df_no_edges)
        self.assertIsInstance(geocode_df, pl.DataFrame)

        # Create a df with edge hexagons (should fail)
        df_with_edges = pl.DataFrame(
            {
                "geocode": pl.Series([1, 2], dtype=pl.UInt64),
                "center": points_series(2),
                "boundary": polygon_series(2),
                "is_edge": pl.Series([True, False], dtype=pl.Boolean),
            }
        )

        # This should raise a validation error
        with self.assertRaises(ValidationError):
            GeocodeNoEdgesSchema.validate(df_with_edges)

    def test_build_geocode_no_edges_df(self):
        """Test building a GeocodeNoEdgesSchema by filtering edges"""
        # Create a GeocodeSchema df with some edge hexagons
        df = pl.DataFrame(
            {
                "geocode": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
                "center": points_series(4),
                "boundary": polygon_series(4),
                "is_edge": pl.Series([True, False, True, False], dtype=pl.Boolean),
            }
        )

        geocode_df = GeocodeSchema.validate(df)
        geocode_lf: dy.LazyFrame[GeocodeSchema] = geocode_df.lazy()

        # Build no-edges df
        no_edges_df = build_geocode_no_edges_lf(geocode_lf).collect()

        # Should only have non-edge geocodes
        self.assertEqual(len(no_edges_df), 2)
        self.assertEqual(set(no_edges_df["geocode"].to_list()), {2, 4})
        self.assertTrue(all(~no_edges_df["is_edge"]))

    def test_index_of_geocode(self):
        """Test the function that finds the index of a geocode in the dataframe"""
        df = pl.DataFrame(
            {
                "geocode": pl.Series(
                    [8514355555555555, 8514355555555557], dtype=pl.UInt64
                ),
                "center": points_series(2),
                "boundary": polygon_series(2),
                "is_edge": pl.Series([True, True], dtype=pl.Boolean),
            }
        )

        geocode_df = GeocodeSchema.validate(df)

        # Test finding index of existing geocode
        index = index_of_geocode(8514355555555555, geocode_df)
        self.assertEqual(index, 0)

        index = index_of_geocode(8514355555555557, geocode_df)
        self.assertEqual(index, 1)

        # Test with non-existent geocode
        with self.assertRaises(ValueError):
            index_of_geocode(8514355555555559, geocode_df)


def points_series(count: int):
    return pl.DataFrame(
        {
            "points_wkt": pl.Series(["POINT(-122.1 37.5)"] * count),
        }
    ).select(pl_st.from_wkt("points_wkt").alias("point"))["point"]


def polygon_series(count: int):
    return pl.DataFrame(
        {
            "polygon_wkt": pl.Series(
                [
                    "POLYGON((-122.1 37.5, -122.1 37.6, -122.0 37.6, -122.0 37.5, -122.1 37.5))"
                ]
                * count
            ),
        }
    ).select(pl_st.from_wkt("polygon_wkt").alias("polygon"))["polygon"]


if __name__ == "__main__":
    unittest.main()
