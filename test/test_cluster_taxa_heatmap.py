"""
Regression tests for the cluster taxa heatmap functionality.

This module contains tests for src/plot/cluster_taxa.py, specifically
regression tests for bugs that have been fixed.
"""

import unittest

import dataframely as dy
import numpy as np
import polars as pl
import polars_st as pl_st
import shapely

from src.constants import KINGDOM_VALUES, TAXON_RANK_VALUES
from src.dataframes.cluster_color import ClusterColorSchema
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesSchema,
)
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsSchema
from src.dataframes.taxonomy import TaxonomySchema
from src.plot.cluster_taxa import create_cluster_taxa_heatmap


class MockGeocodeDistanceMatrix:
    """Mock distance matrix for testing."""

    def __init__(self, n_geocodes: int):
        # Create a simple condensed distance matrix
        # Number of elements in condensed form: n*(n-1)/2
        n_elements = n_geocodes * (n_geocodes - 1) // 2
        self._condensed = np.random.rand(n_elements)

    def condensed(self) -> np.ndarray:
        return self._condensed


class TestClusterTaxaHeatmap(unittest.TestCase):
    """Tests for create_cluster_taxa_heatmap function."""

    def test_geocode_without_taxa_data_no_division_by_zero(self):
        """
        Regression test: Ensure no ZeroDivisionError when a geocode exists
        in geocode_lf but has no corresponding data in geocode_taxa_counts_lf.

        This was a bug where iterating over all geocodes from geocode_lf would
        cause a division by zero when computing geocode_average for geocodes
        that had no taxa counts (geocode_counts_all == 0).

        The fix filters ordered_geocodes to only include geocodes that have
        data in geocode_taxa_counts_lf.
        """
        # Create geocode_lf with 3 geocodes, but only 2 will have taxa data
        centers = [
            shapely.Point(-70.0, 42.0),
            shapely.Point(-70.1, 42.0),
            shapely.Point(-70.2, 42.0),  # This geocode will have NO taxa data
        ]
        boundaries = []
        for center in centers:
            x, y = center.x, center.y
            hex_points = [
                (x + 0.01, y),
                (x + 0.005, y + 0.01),
                (x - 0.005, y + 0.01),
                (x - 0.01, y),
                (x - 0.005, y - 0.01),
                (x + 0.005, y - 0.01),
            ]
            boundaries.append(shapely.Polygon(hex_points))

        geocode_df = pl.DataFrame(
            {
                "geocode": [1000, 2000, 3000],  # 3 geocodes
                "is_edge": [False, False, False],
            }
        ).with_columns(
            pl.col("geocode").cast(pl.UInt64),
            pl.col("is_edge").cast(pl.Boolean),
        )
        geocode_df = geocode_df.with_columns(
            pl_st.from_shapely(pl.Series(centers)).alias("center"),
            pl_st.from_shapely(pl.Series(boundaries)).alias("boundary"),
        )
        geocode_lf = GeocodeNoEdgesSchema.validate(geocode_df, eager=False)

        # Create geocode_taxa_counts with data for only 2 of the 3 geocodes
        # Geocode 3000 intentionally has NO taxa data
        geocode_taxa_counts_df = pl.DataFrame(
            {
                "geocode": [1000, 1000, 2000, 2000],  # No data for geocode 3000!
                "taxonId": [101, 102, 101, 102],
                "count": [10, 5, 8, 12],
            }
        ).with_columns(
            pl.col("geocode").cast(pl.UInt64),
            pl.col("taxonId").cast(pl.UInt32),
            pl.col("count").cast(pl.UInt32),
        )
        geocode_taxa_counts_lf = GeocodeTaxaCountsSchema.validate(
            geocode_taxa_counts_df, eager=False
        )

        # Create geocode_cluster_df for all 3 geocodes
        geocode_cluster_df = GeocodeClusterSchema.validate(
            pl.DataFrame(
                {
                    "geocode": [1000, 2000, 3000],
                    "cluster": [0, 0, 1],
                }
            ).with_columns(
                pl.col("geocode").cast(pl.UInt64),
                pl.col("cluster").cast(pl.UInt32),
            )
        )

        # Create cluster_colors_df
        cluster_colors_df = ClusterColorSchema.validate(
            pl.DataFrame(
                {
                    "cluster": [0, 1],
                    "color": ["#ff0000", "#0000ff"],
                    "darkened_color": ["#800000", "#000080"],
                }
            ).with_columns(pl.col("cluster").cast(pl.UInt32))
        )

        # Create mock distance matrix for 2 geocodes (those with taxa data)
        geocode_distance_matrix = MockGeocodeDistanceMatrix(n_geocodes=2)

        # Create cluster_significant_differences_df with taxonId 101
        significant_diff_df = pl.DataFrame(
            {
                "cluster": [0],
                "taxonId": [101],
                "p_value": [0.01],
                "log2_fold_change": [1.5],
                "cluster_count": [18],
                "neighbor_count": [5],
                "high_log2_high_count_score": [0.5],
                "low_log2_high_count_score": [0.0],
            }
        ).with_columns(
            pl.col("cluster").cast(pl.UInt32),
            pl.col("taxonId").cast(pl.UInt32),
            pl.col("cluster_count").cast(pl.UInt32),
            pl.col("neighbor_count").cast(pl.UInt32),
        )
        cluster_significant_differences_df = (
            ClusterSignificantDifferencesSchema.validate(significant_diff_df)
        )

        # Create taxonomy_df
        taxonomy_df = pl.DataFrame(
            {
                "taxonId": [101, 102],
                "kingdom": ["Animalia", "Animalia"],
                "phylum": ["Chordata", "Chordata"],
                "class": ["Mammalia", "Aves"],
                "order": ["Carnivora", "Passeriformes"],
                "family": ["Felidae", "Corvidae"],
                "genus": ["Panthera", "Corvus"],
                "species": ["leo", "corax"],
                "taxonRank": ["SPECIES", "SPECIES"],
                "scientificName": ["Panthera leo", "Corvus corax"],
                "gbifTaxonId": [5219404, 2482468],
            }
        ).with_columns(
            pl.col("taxonId").cast(pl.UInt32),
            pl.col("kingdom").cast(pl.Enum(KINGDOM_VALUES)),
            pl.col("phylum").cast(pl.Categorical),
            pl.col("class").cast(pl.Categorical),
            pl.col("order").cast(pl.Categorical),
            pl.col("family").cast(pl.Categorical),
            pl.col("genus").cast(pl.Categorical),
            pl.col("taxonRank").cast(pl.Enum(TAXON_RANK_VALUES)),
            pl.col("gbifTaxonId").cast(pl.UInt32),
        )
        taxonomy_df = TaxonomySchema.validate(taxonomy_df)

        # Create cluster_taxa_statistics_df
        cluster_taxa_stats_df = pl.DataFrame(
            {
                "cluster": [0, 0, 1, 1, None, None],
                "taxonId": [101, 102, 101, 102, 101, 102],
                "count": [18, 17, 0, 0, 18, 17],
                "average": [0.514, 0.486, 0.0, 0.0, 0.514, 0.486],
            },
            schema={
                "cluster": pl.UInt32(),
                "taxonId": pl.UInt32(),
                "count": pl.UInt32(),
                "average": pl.Float64(),
            },
        )
        cluster_taxa_statistics_df = ClusterTaxaStatisticsSchema.validate(
            cluster_taxa_stats_df
        )

        # This should NOT raise a ZeroDivisionError
        # Before the fix, this would fail because geocode 3000 has no taxa data
        try:
            result = create_cluster_taxa_heatmap(
                geocode_lf=geocode_lf,
                geocode_cluster_df=geocode_cluster_df,
                cluster_colors_df=cluster_colors_df,
                geocode_distance_matrix=geocode_distance_matrix,
                cluster_significant_differences_df=cluster_significant_differences_df,
                taxonomy_df=taxonomy_df,
                geocode_taxa_counts_lf=geocode_taxa_counts_lf,
                cluster_taxa_statistics_df=cluster_taxa_statistics_df,
                limit_species=1,
            )
            # Result can be None if no significant differences, or a clustermap
            # Either way, no exception means the test passes
        except ZeroDivisionError:
            self.fail(
                "create_cluster_taxa_heatmap raised ZeroDivisionError when a geocode "
                "in geocode_lf had no data in geocode_taxa_counts_lf"
            )

    def test_all_geocodes_have_taxa_data(self):
        """
        Test normal case where all geocodes in geocode_lf have taxa data.
        This ensures the fix doesn't break the normal use case.
        """
        # Create geocode_lf with 2 geocodes, both with taxa data
        centers = [
            shapely.Point(-70.0, 42.0),
            shapely.Point(-70.1, 42.0),
        ]
        boundaries = []
        for center in centers:
            x, y = center.x, center.y
            hex_points = [
                (x + 0.01, y),
                (x + 0.005, y + 0.01),
                (x - 0.005, y + 0.01),
                (x - 0.01, y),
                (x - 0.005, y - 0.01),
                (x + 0.005, y - 0.01),
            ]
            boundaries.append(shapely.Polygon(hex_points))

        geocode_df = pl.DataFrame(
            {
                "geocode": [1000, 2000],
                "is_edge": [False, False],
            }
        ).with_columns(
            pl.col("geocode").cast(pl.UInt64),
            pl.col("is_edge").cast(pl.Boolean),
        )
        geocode_df = geocode_df.with_columns(
            pl_st.from_shapely(pl.Series(centers)).alias("center"),
            pl_st.from_shapely(pl.Series(boundaries)).alias("boundary"),
        )
        geocode_lf = GeocodeNoEdgesSchema.validate(geocode_df, eager=False)

        # Create geocode_taxa_counts with data for both geocodes
        geocode_taxa_counts_df = pl.DataFrame(
            {
                "geocode": [1000, 1000, 2000, 2000],
                "taxonId": [101, 102, 101, 102],
                "count": [10, 5, 8, 12],
            }
        ).with_columns(
            pl.col("geocode").cast(pl.UInt64),
            pl.col("taxonId").cast(pl.UInt32),
            pl.col("count").cast(pl.UInt32),
        )
        geocode_taxa_counts_lf = GeocodeTaxaCountsSchema.validate(
            geocode_taxa_counts_df, eager=False
        )

        # Create geocode_cluster_df
        geocode_cluster_df = GeocodeClusterSchema.validate(
            pl.DataFrame(
                {
                    "geocode": [1000, 2000],
                    "cluster": [0, 1],
                }
            ).with_columns(
                pl.col("geocode").cast(pl.UInt64),
                pl.col("cluster").cast(pl.UInt32),
            )
        )

        # Create cluster_colors_df
        cluster_colors_df = ClusterColorSchema.validate(
            pl.DataFrame(
                {
                    "cluster": [0, 1],
                    "color": ["#ff0000", "#0000ff"],
                    "darkened_color": ["#800000", "#000080"],
                }
            ).with_columns(pl.col("cluster").cast(pl.UInt32))
        )

        # Create mock distance matrix
        geocode_distance_matrix = MockGeocodeDistanceMatrix(n_geocodes=2)

        # Create cluster_significant_differences_df with taxonId 101
        significant_diff_df = pl.DataFrame(
            {
                "cluster": [0],
                "taxonId": [101],
                "p_value": [0.01],
                "log2_fold_change": [1.5],
                "cluster_count": [10],
                "neighbor_count": [8],
                "high_log2_high_count_score": [0.5],
                "low_log2_high_count_score": [0.0],
            }
        ).with_columns(
            pl.col("cluster").cast(pl.UInt32),
            pl.col("taxonId").cast(pl.UInt32),
            pl.col("cluster_count").cast(pl.UInt32),
            pl.col("neighbor_count").cast(pl.UInt32),
        )
        cluster_significant_differences_df = (
            ClusterSignificantDifferencesSchema.validate(significant_diff_df)
        )

        # Create taxonomy_df
        taxonomy_df = pl.DataFrame(
            {
                "taxonId": [101, 102],
                "kingdom": ["Animalia", "Animalia"],
                "phylum": ["Chordata", "Chordata"],
                "class": ["Mammalia", "Aves"],
                "order": ["Carnivora", "Passeriformes"],
                "family": ["Felidae", "Corvidae"],
                "genus": ["Panthera", "Corvus"],
                "species": ["leo", "corax"],
                "taxonRank": ["SPECIES", "SPECIES"],
                "scientificName": ["Panthera leo", "Corvus corax"],
                "gbifTaxonId": [5219404, 2482468],
            }
        ).with_columns(
            pl.col("taxonId").cast(pl.UInt32),
            pl.col("kingdom").cast(pl.Enum(KINGDOM_VALUES)),
            pl.col("phylum").cast(pl.Categorical),
            pl.col("class").cast(pl.Categorical),
            pl.col("order").cast(pl.Categorical),
            pl.col("family").cast(pl.Categorical),
            pl.col("genus").cast(pl.Categorical),
            pl.col("taxonRank").cast(pl.Enum(TAXON_RANK_VALUES)),
            pl.col("gbifTaxonId").cast(pl.UInt32),
        )
        taxonomy_df = TaxonomySchema.validate(taxonomy_df)

        # Create cluster_taxa_statistics_df
        cluster_taxa_stats_df = pl.DataFrame(
            {
                "cluster": [0, 0, 1, 1, None, None],
                "taxonId": [101, 102, 101, 102, 101, 102],
                "count": [10, 5, 8, 12, 18, 17],
                "average": [0.667, 0.333, 0.4, 0.6, 0.514, 0.486],
            },
            schema={
                "cluster": pl.UInt32(),
                "taxonId": pl.UInt32(),
                "count": pl.UInt32(),
                "average": pl.Float64(),
            },
        )
        cluster_taxa_statistics_df = ClusterTaxaStatisticsSchema.validate(
            cluster_taxa_stats_df
        )

        # This should work without any errors
        result = create_cluster_taxa_heatmap(
            geocode_lf=geocode_lf,
            geocode_cluster_df=geocode_cluster_df,
            cluster_colors_df=cluster_colors_df,
            geocode_distance_matrix=geocode_distance_matrix,
            cluster_significant_differences_df=cluster_significant_differences_df,
            taxonomy_df=taxonomy_df,
            geocode_taxa_counts_lf=geocode_taxa_counts_lf,
            cluster_taxa_statistics_df=cluster_taxa_statistics_df,
            limit_species=1,
        )
        # Result should be a clustermap (not None since we have significant differences)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
