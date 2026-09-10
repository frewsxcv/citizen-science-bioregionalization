import unittest

import polars as pl

from src.dataframes.geocode_taxa_counts import filter_top_taxa_lf


class TestFilterTopTaxaLf(unittest.TestCase):
    """Tests for the filter_top_taxa_lf function."""

    def _create_test_data(self) -> pl.LazyFrame:
        """
        Creates test data with known taxa distributions.

        Taxa distribution:
        - taxonId 0: present in 3 geocodes (1000, 2000, 3000), total count = 10
        - taxonId 1: present in 2 geocodes (1000, 2000), total count = 6
        - taxonId 2: present in 2 geocodes (2000, 3000), total count = 12
        - taxonId 3: present in 1 geocode (3000), total count = 3

        Geocodes: 1000, 2000, 3000 (3 total)
        """
        data = [
            {"geocode": 1000, "taxonId": 0, "count": 5},
            {"geocode": 1000, "taxonId": 1, "count": 4},
            {"geocode": 2000, "taxonId": 0, "count": 3},
            {"geocode": 2000, "taxonId": 1, "count": 2},
            {"geocode": 2000, "taxonId": 2, "count": 8},
            {"geocode": 3000, "taxonId": 0, "count": 2},
            {"geocode": 3000, "taxonId": 2, "count": 4},
            {"geocode": 3000, "taxonId": 3, "count": 3},
        ]
        df = pl.DataFrame(data).with_columns(
            pl.col("geocode").cast(pl.UInt64),
            pl.col("taxonId").cast(pl.UInt32),
            pl.col("count").cast(pl.UInt32),
        )
        return df.lazy()

    def test_no_filtering_when_params_none(self):
        """Test that no filtering occurs when both parameters are None."""
        lf = self._create_test_data()
        result = filter_top_taxa_lf(lf, max_taxa=None, min_geocode_presence=None)
        result_df = result.collect()

        # Should have all 8 rows
        self.assertEqual(result_df.height, 8)
        # Should have all 4 taxa
        self.assertEqual(result_df["taxonId"].n_unique(), 4)

    def test_filter_by_max_taxa(self):
        """Test filtering to top N taxa by total count."""
        lf = self._create_test_data()

        # Filter to top 2 taxa by count
        # taxonId 2 has total 12, taxonId 0 has total 10
        result = filter_top_taxa_lf(lf, max_taxa=2, min_geocode_presence=None)
        result_df = result.collect()

        unique_taxa = result_df["taxonId"].unique().sort().to_list()
        self.assertEqual(unique_taxa, [0, 2])

    def test_filter_by_max_taxa_single(self):
        """Test filtering to single top taxon."""
        lf = self._create_test_data()

        # Filter to top 1 taxon by count (taxonId 2 with count 12)
        result = filter_top_taxa_lf(lf, max_taxa=1, min_geocode_presence=None)
        result_df = result.collect()

        unique_taxa = result_df["taxonId"].unique().to_list()
        self.assertEqual(unique_taxa, [2])

    def test_filter_by_min_geocode_presence(self):
        """Test filtering by minimum geocode presence fraction."""
        lf = self._create_test_data()

        # 3 geocodes total. 0.67 presence = int(3 * 0.67) = 2 geocodes minimum
        # taxonId 0: 3 geocodes (keep)
        # taxonId 1: 2 geocodes (keep)
        # taxonId 2: 2 geocodes (keep)
        # taxonId 3: 1 geocode (filter out)
        result = filter_top_taxa_lf(lf, max_taxa=None, min_geocode_presence=0.67)
        result_df = result.collect()

        unique_taxa = result_df["taxonId"].unique().sort().to_list()
        self.assertEqual(unique_taxa, [0, 1, 2])

    def test_filter_by_high_min_geocode_presence(self):
        """Test filtering with high presence threshold."""
        lf = self._create_test_data()

        # 3 geocodes total. 1.0 presence = int(3 * 1.0) = 3 geocodes minimum
        # Only taxonId 0 is present in all 3 geocodes
        result = filter_top_taxa_lf(lf, max_taxa=None, min_geocode_presence=1.0)
        result_df = result.collect()

        unique_taxa = result_df["taxonId"].unique().to_list()
        self.assertEqual(unique_taxa, [0])

    def test_filter_combined(self):
        """Test combining both filters."""
        lf = self._create_test_data()

        # First filter by presence (>= 0.67 -> int(3*0.67)=2 geocodes): keeps taxa 0, 1, 2
        # Then filter to top 2 by count: taxa 0 (10) and 2 (12)
        result = filter_top_taxa_lf(lf, max_taxa=2, min_geocode_presence=0.67)
        result_df = result.collect()

        unique_taxa = result_df["taxonId"].unique().sort().to_list()
        self.assertEqual(unique_taxa, [0, 2])

    def test_preserves_schema(self):
        """Test that the filtered result still conforms to the schema."""
        lf = self._create_test_data()
        result = filter_top_taxa_lf(lf, max_taxa=2, min_geocode_presence=0.3)

        # Should not raise validation errors
        result_df = result.collect()

        # Verify column types
        self.assertEqual(result_df.schema["geocode"], pl.UInt64)
        self.assertEqual(result_df.schema["taxonId"], pl.UInt32)
        self.assertEqual(result_df.schema["count"], pl.UInt32)

    def test_preserves_counts(self):
        """Test that original count values are preserved after filtering."""
        lf = self._create_test_data()
        result = filter_top_taxa_lf(lf, max_taxa=1, min_geocode_presence=None)
        result_df = result.collect()

        # taxonId 2 should be kept with original counts
        geocode_2000_count = result_df.filter(pl.col("geocode") == 2000)["count"].item()
        geocode_3000_count = result_df.filter(pl.col("geocode") == 3000)["count"].item()

        self.assertEqual(geocode_2000_count, 8)
        self.assertEqual(geocode_3000_count, 4)


if __name__ == "__main__":
    unittest.main()


class TestMaxTaxaTieBreaking(unittest.TestCase):
    """The top-N cut must not depend on the order rows arrive in.

    Regression test for a real change in the published map. Two runs on
    identical input kept the same count of taxa (10000 of 21432) but a
    different *set* of them -- 1598202 rows against 1598224 -- because the cut
    at max_taxa landed among taxa tied on total_count, and `sort` on that one
    key left their relative order to whatever group_by emitted. Silhouette
    moved from 0.3264 to 0.3374 on nothing but hash order.
    """

    def _tied_data(self, taxon_order: list[int]) -> pl.LazyFrame:
        """One clear winner plus four taxa deliberately tied at the cut."""
        rows = [{"geocode": 1000, "taxonId": 1, "count": 100}]
        for taxon_id in taxon_order:
            rows.append({"geocode": 2000, "taxonId": taxon_id, "count": 5})
        df = pl.DataFrame(rows).with_columns(
            pl.col("geocode").cast(pl.UInt64),
            pl.col("taxonId").cast(pl.UInt32),
            pl.col("count").cast(pl.UInt32),
        )
        return df.lazy()

    def _top_taxa(self, taxon_order: list[int]) -> list[int]:
        result = filter_top_taxa_lf(
            self._tied_data(taxon_order), max_taxa=3, min_geocode_presence=None
        ).collect()
        return sorted(result["taxonId"].unique().to_list())

    def test_ties_broken_by_taxon_id(self):
        """Among equals, the lowest taxonIds win -- a defined answer, not a race."""
        # taxonId 1 leads on count (100). Ten, 11, 12 and 13 all total 5, and
        # max_taxa=3 admits exactly two of them.
        self.assertEqual(self._top_taxa([10, 11, 12, 13]), [1, 10, 11])

    def test_row_order_does_not_change_the_selection(self):
        """Permuting the input must not change which taxa survive."""
        orders = [
            [10, 11, 12, 13],
            [13, 12, 11, 10],
            [12, 10, 13, 11],
            [11, 13, 10, 12],
        ]
        selections = [self._top_taxa(order) for order in orders]
        for order, selection in zip(orders, selections):
            self.assertEqual(
                selection,
                selections[0],
                f"input order {order} produced {selection}, "
                f"but {orders[0]} produced {selections[0]}",
            )
