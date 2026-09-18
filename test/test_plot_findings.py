"""
Unit tests for the findings charts.

Every bug these cover was silent in the notebook: the chart rendered, with its
title, axes and legend intact, and simply drew no marks. Nothing raised, so the
export looked healthy. The assertions here are therefore about the *encoded
data* -- what Vega will actually be handed -- rather than about the spec
building without error.
"""

import unittest

import numpy as np
import polars as pl

from src.plot.findings import cluster_geography, dissimilarity_vs_effort, metrics_by_k


def _chart_data(spec: dict) -> list[dict]:
    """The rows Vega will be handed.

    Altair hoists the values into a named dataset, so reading `chart.data` back
    inspects the input rather than the encoded output -- which is where each of
    these bugs lived.
    """
    return spec["datasets"][spec["data"]["name"]]


def _layer_data(spec: dict, layer: int) -> list[dict]:
    return spec["datasets"][spec["layer"][layer]["data"]["name"]]


def mock_cluster_metrics_df(k_values: list[int] | None = None) -> pl.DataFrame:
    """A metrics frame with the column names build_cluster_metrics_df emits."""
    if k_values is None:
        k_values = [2, 3, 4, 5]
    return pl.DataFrame(
        {
            "rank": list(range(1, len(k_values) + 1)),
            "num_clusters": k_values,
            "silhouette_score": [0.5 - 0.05 * i for i in range(len(k_values))],
            "calinski_harabasz_score": [100.0 - 10 * i for i in range(len(k_values))],
            "davies_bouldin_score": [1.0 + 0.1 * i for i in range(len(k_values))],
            "inertia": [500.0 - 50 * i for i in range(len(k_values))],
            "combined_score": [0.8 - 0.05 * i for i in range(len(k_values))],
        }
    )


class TestMetricsByK(unittest.TestCase):
    def test_plots_every_measure_the_metrics_frame_carries(self) -> None:
        """The measure columns are read under the names the builder emits.

        They were previously read without the `_score` suffix, so three of the
        four never matched, and only `combined_score` was ever drawn.
        """
        spec = metrics_by_k(mock_cluster_metrics_df(), chosen_k=2).to_dict()

        drawn = {row["metric"] for row in _layer_data(spec, 0)}
        self.assertEqual(len(drawn), 4, f"expected four measures, drew {drawn}")

    def test_legend_names_only_measures_that_are_drawn(self) -> None:
        """A fixed domain let the legend advertise a line that did not exist."""
        df = mock_cluster_metrics_df().drop("calinski_harabasz_score")
        spec = metrics_by_k(df, chosen_k=2).to_dict()

        domain = set(spec["layer"][0]["encoding"]["color"]["scale"]["domain"])
        self.assertEqual(domain, {row["metric"] for row in _layer_data(spec, 0)})

    def test_every_measure_gets_its_own_colour(self) -> None:
        spec = metrics_by_k(mock_cluster_metrics_df(), chosen_k=2).to_dict()
        scale = spec["layer"][0]["encoding"]["color"]["scale"]
        self.assertEqual(len(scale["range"]), len(scale["domain"]))
        self.assertEqual(len(set(scale["range"])), len(scale["range"]))


class TestClusterGeography(unittest.TestCase):
    def _frames(self) -> tuple[pl.DataFrame, pl.LazyFrame, pl.DataFrame]:
        # Two real H3 cells so polars_h3 can resolve centres.
        geocodes = [608448695024746495, 608526105032261631]
        geocode_cluster_df = pl.DataFrame(
            {"geocode": pl.Series(geocodes, dtype=pl.UInt64), "cluster": [0, 1]}
        )
        geocode_lf = pl.DataFrame(
            {"geocode": pl.Series(geocodes, dtype=pl.UInt64)}
        ).lazy()
        cluster_colors_df = pl.DataFrame(
            {"cluster": [0, 1], "color": ["#aa7744", "#cc4433"]}
        )
        return geocode_cluster_df, geocode_lf, cluster_colors_df

    def test_cluster_values_match_the_colour_scale_domain(self) -> None:
        """A numeric datum never matches a string domain.

        The legend reads the domain rather than the data, so this failed as a
        fully drawn chart with no points in it.
        """
        geocode_cluster_df, geocode_lf, cluster_colors_df = self._frames()
        spec = cluster_geography(
            geocode_cluster_df, geocode_lf, cluster_colors_df
        ).to_dict()

        domain = set(spec["encoding"]["color"]["scale"]["domain"])
        encoded = {row["cluster"] for row in _chart_data(spec)}
        self.assertTrue(
            all(isinstance(c, str) for c in encoded),
            f"cluster values {encoded} are not strings",
        )
        self.assertTrue(
            encoded <= domain,
            f"cluster values {encoded} are not in the scale domain {domain}",
        )

    def test_geocodes_are_encoded_as_text(self) -> None:
        """u64 cell ids exceed JavaScript's safe integer range."""
        geocode_cluster_df, geocode_lf, cluster_colors_df = self._frames()
        spec = cluster_geography(
            geocode_cluster_df, geocode_lf, cluster_colors_df
        ).to_dict()

        self.assertTrue(
            all(isinstance(row["geocode"], str) for row in _chart_data(spec))
        )

    def test_works_without_a_colour_frame(self) -> None:
        geocode_cluster_df, geocode_lf, _ = self._frames()
        spec = cluster_geography(geocode_cluster_df, geocode_lf, None).to_dict()
        self.assertEqual(len(_chart_data(spec)), 2)


class TestDissimilarityVsEffort(unittest.TestCase):
    def _counts_lf(self, n: int) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "geocode": list(range(n)),
                "taxonId": [0] * n,
                "count": [int(10 ** (i / 20)) for i in range(n)],
            }
        ).lazy()

    def test_bins_hold_the_medians_they_report(self) -> None:
        n = 60
        condensed = np.ones(n * (n - 1) // 2)
        chart, _rho = dissimilarity_vs_effort(condensed, self._counts_lf(n))

        assert chart is not None
        self.assertTrue((chart.data["median"] == 1.0).all())
        self.assertEqual(chart.data["pairs"].sum(), n * (n - 1) // 2)

    def test_returns_no_chart_when_no_bin_has_enough_pairs(self) -> None:
        """An empty chart would claim the question had been answered."""
        n = 3
        condensed = np.ones(n * (n - 1) // 2)
        chart, _rho = dissimilarity_vs_effort(condensed, self._counts_lf(n))
        self.assertIsNone(chart)

    def test_rejects_a_distance_matrix_of_the_wrong_size(self) -> None:
        with self.assertRaises(ValueError):
            dissimilarity_vs_effort(np.ones(5), self._counts_lf(60))


if __name__ == "__main__":
    unittest.main()
