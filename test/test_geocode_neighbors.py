import unittest

import dataframely as dy
import networkx as nx
import polars as pl
import polars_st as pl_st

from src.dataframes.geocode_neighbors import (
    GeocodeNeighborsSchema,
    _add_indirect_neighbor_edge,
    _df_to_graph,
    build_geocode_neighbors_df,
    build_geocode_neighbors_no_edges_df,
    graph,
)


class TestGeocodeNeighborsSchema(unittest.TestCase):
    def test_geocode_neighbors_schema(self):
        """Test that the GeocodeNeighborsSchema validates its schema correctly"""
        df = pl.DataFrame(
            {
                "geocode": pl.Series(
                    [0x8514355555555555, 0x8514355555555557], dtype=pl.UInt64
                ),
                "direct_neighbors": pl.Series(
                    [[0x8514355555555557], [0x8514355555555555]],
                    dtype=pl.List(pl.UInt64),
                ),
                "direct_and_indirect_neighbors": pl.Series(
                    [[0x8514355555555557], [0x8514355555555555]],
                    dtype=pl.List(pl.UInt64),
                ),
            }
        )

        # This should not raise an exception
        neighbors_df = GeocodeNeighborsSchema.validate(df)
        self.assertIsInstance(neighbors_df, pl.DataFrame)

    def test_graph_conversion(self):
        """Test conversion of GeocodeNeighborsSchema to a graph"""
        df = pl.DataFrame(
            {
                "geocode": pl.Series([1, 2, 3], dtype=pl.UInt64),
                "direct_neighbors": pl.Series(
                    [
                        [2],  # 1 connected to 2
                        [1, 3],  # 2 connected to 1 and 3
                        [2],  # 3 connected to 2
                    ],
                    dtype=pl.List(pl.UInt64),
                ),
                "direct_and_indirect_neighbors": pl.Series(
                    [[2], [1, 3], [2]], dtype=pl.List(pl.UInt64)
                ),
            }
        )

        neighbors_df = GeocodeNeighborsSchema.validate(df)

        # Convert to graph
        g = graph(neighbors_df)

        # Check that the graph has the right structure
        self.assertEqual(len(g.nodes), 3)
        self.assertEqual(len(g.edges), 2)  # 1-2 and 2-3

        # Check connectivity
        self.assertTrue(g.has_edge(1, 2))
        self.assertTrue(g.has_edge(2, 3))
        self.assertFalse(g.has_edge(1, 3))

        # Check connected components
        self.assertEqual(nx.number_connected_components(g), 1)

    def test_graph_with_indirect_neighbors(self):
        """Test graph conversion includes indirect neighbors when requested"""
        df = pl.DataFrame(
            {
                "geocode": pl.Series([1, 2, 3], dtype=pl.UInt64),
                "direct_neighbors": pl.Series(
                    [[2], [1], []],
                    dtype=pl.List(pl.UInt64),  # 3 has no direct neighbors
                ),
                "direct_and_indirect_neighbors": pl.Series(
                    [
                        [2, 3],  # 1 indirectly connected to 3
                        [1],
                        [1],  # 3 indirectly connected to 1
                    ],
                    dtype=pl.List(pl.UInt64),
                ),
            }
        )

        neighbors_df = GeocodeNeighborsSchema.validate(df)

        # Without indirect neighbors, 3 is disconnected
        g_direct = graph(neighbors_df, include_indirect_neighbors=False)
        self.assertEqual(nx.number_connected_components(g_direct), 2)

        # With indirect neighbors, all connected
        g_indirect = graph(neighbors_df, include_indirect_neighbors=True)
        self.assertEqual(nx.number_connected_components(g_indirect), 1)
        self.assertTrue(g_indirect.has_edge(1, 3))

    def test_reduce_connected_components(self):
        """Test that _reduce_connected_components_to_one works via _df_to_graph"""
        # Create a dataframe with multiple disconnected components
        df = pl.DataFrame(
            {
                "geocode": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
                "center": points_series(4),
                "direct_neighbors": pl.Series(
                    [
                        [2],  # 1 connected to 2
                        [1],  # 2 connected to 1
                        [4],  # 3 connected to 4
                        [3],  # 4 connected to 3
                    ],
                    dtype=pl.List(pl.UInt64),
                ),
                "direct_and_indirect_neighbors": pl.Series(
                    [[2], [1], [4], [3]], dtype=pl.List(pl.UInt64)
                ),
            }
        )

        # Initially we have two components
        g = _df_to_graph(df)
        self.assertEqual(nx.number_connected_components(g), 2)

        # Manually simulate connecting the components
        df_with_connection = df.with_columns(
            direct_and_indirect_neighbors=pl.when(pl.col("geocode") == 1)
            .then(pl.lit([[2, 3]], dtype=pl.List(pl.List(pl.UInt64))).list.first())
            .when(pl.col("geocode") == 3)
            .then(pl.lit([[4, 1]], dtype=pl.List(pl.List(pl.UInt64))).list.first())
            .otherwise(pl.col("direct_and_indirect_neighbors"))
        )

        # Now we should have one component when using the direct_and_indirect_neighbors
        combined_graph = _df_to_graph(
            df_with_connection, include_indirect_neighbors=True
        )
        self.assertEqual(nx.number_connected_components(combined_graph), 1)

        # Check that there are now paths between nodes in different original components
        self.assertTrue(
            nx.has_path(combined_graph, 1, 3),
            "There should be a path from node 1 to node 3",
        )

    def test_add_indirect_neighbor_edge_preserves_uint64_precision(self):
        """Regression test: ensure large H3 geocodes don't lose precision when adding indirect neighbors.

        This test verifies the fix for a bug where `list.concat([python_int])` caused Polars
        to infer float64 type, losing precision for large 64-bit integers. H3 cell IDs like
        595148721944002559 would be rounded to 595148721944002560 (off by 1).
        """
        # These are real H3 cell IDs that exhibited the bug
        geocode1 = 595148721944002559
        geocode2 = 595148996821909504

        df = pl.DataFrame(
            {
                "geocode": pl.Series([geocode1, geocode2], dtype=pl.UInt64),
                "direct_neighbors": pl.Series(
                    [[595685549906329599], [595685549906329599]],
                    dtype=pl.List(pl.UInt64),
                ),
                "direct_and_indirect_neighbors": pl.Series(
                    [[595685549906329599], [595685549906329599]],
                    dtype=pl.List(pl.UInt64),
                ),
            }
        )

        # Add an indirect neighbor edge between the two geocodes
        result_df = _add_indirect_neighbor_edge(df, geocode1, geocode2)

        # Verify geocode1's neighbors now include geocode2 (exact value, not off by 1)
        geocode1_neighbors = result_df.filter(pl.col("geocode") == geocode1)[
            "direct_and_indirect_neighbors"
        ][0]
        self.assertIn(
            geocode2,
            geocode1_neighbors,
            f"Expected exact geocode {geocode2} in neighbors, got {geocode1_neighbors}. "
            f"If {geocode2 + 1} is present instead, there's a float64 precision loss bug.",
        )

        # Verify geocode2's neighbors now include geocode1 (exact value, not off by 1)
        geocode2_neighbors = result_df.filter(pl.col("geocode") == geocode2)[
            "direct_and_indirect_neighbors"
        ][0]
        self.assertIn(
            geocode1,
            geocode2_neighbors,
            f"Expected exact geocode {geocode1} in neighbors, got {geocode2_neighbors}. "
            f"If {geocode1 + 1} is present instead, there's a float64 precision loss bug.",
        )

        # Also verify the wrong values are NOT present (the off-by-one values)
        self.assertNotIn(
            geocode2 + 1,
            geocode1_neighbors,
            "Found off-by-one geocode value, indicating float64 precision loss",
        )
        self.assertNotIn(
            geocode1 + 1,
            geocode2_neighbors,
            "Found off-by-one geocode value, indicating float64 precision loss",
        )

    def test_build_geocode_neighbors_no_edges_df(self):
        """Test building neighbor relationships for non-edge geocodes"""
        # Create initial neighbors df
        neighbors_df = GeocodeNeighborsSchema.validate(
            pl.DataFrame(
                {
                    "geocode": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
                    "direct_neighbors": pl.Series(
                        [[2, 3], [1, 4], [1, 4], [2, 3]], dtype=pl.List(pl.UInt64)
                    ),
                    "direct_and_indirect_neighbors": pl.Series(
                        [[2, 3], [1, 4], [1, 4], [2, 3]], dtype=pl.List(pl.UInt64)
                    ),
                }
            )
        )

        # Create non-edge geocodes df (only geocodes 2 and 4)
        no_edges_df = pl.DataFrame(
            {
                "geocode": pl.Series([2, 4], dtype=pl.UInt64),
                "center": points_series(2),
            }
        )

        # Build filtered neighbors
        result = build_geocode_neighbors_no_edges_df(neighbors_df, no_edges_df)

        # Should only have geocodes 2 and 4
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["geocode"].to_list()), {2, 4})

        # Neighbor lists should only reference valid geocodes
        for neighbors in result["direct_neighbors"].to_list():
            for neighbor in neighbors:
                self.assertIn(neighbor, {2, 4})

        for neighbors in result["direct_and_indirect_neighbors"].to_list():
            for neighbor in neighbors:
                self.assertIn(neighbor, {2, 4})


def points_series(count: int):
    return pl.DataFrame(
        {
            "points_wkt": pl.Series(["POINT(-122.1 37.5)"] * count),
        }
    ).select(pl_st.from_wkt("points_wkt").alias("point"))["point"]


if __name__ == "__main__":
    unittest.main()
