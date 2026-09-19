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

from src.plot.findings import dissimilarity_vs_effort, metrics_by_k


def _chart_data(spec: dict) -> list[dict]:
    """The rows Vega will be handed.

    Altair hoists the values into a named dataset, so reading `chart.data` back
    inspects the input rather than the encoded output -- which is where each of
    these bugs lived.
    """
    return spec["datasets"][spec["data"]["name"]]


def _layer_data(spec: dict, layer: int) -> list[dict]:
    """Distinct rows for one layer, flattening the nesting Altair emits.

    The cut markers are a layer of their own -- a rule plus its label, per cut
    -- and Altair hoists the data to whichever level shares it, so a nested
    entry may carry `data` at the outer level, the inner level, or both.
    """

    def walk(entry: dict, inherited: dict | None) -> list[dict]:
        data = entry.get("data", inherited)
        if "layer" in entry:
            rows = []
            for inner in entry["layer"]:
                for row in walk(inner, data):
                    if row not in rows:
                        rows.append(row)
            return rows
        assert data is not None, "layer carries no data at any level"
        return spec["datasets"][data["name"]]

    return walk(spec["layer"][layer], spec.get("data"))


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
        spec = metrics_by_k(mock_cluster_metrics_df(), published_k=4, selector_k=2).to_dict()

        drawn = {row["metric"] for row in _layer_data(spec, 0)}
        self.assertEqual(len(drawn), 4, f"expected four measures, drew {drawn}")

    def test_legend_names_only_measures_that_are_drawn(self) -> None:
        """A fixed domain let the legend advertise a line that did not exist."""
        df = mock_cluster_metrics_df().drop("calinski_harabasz_score")
        spec = metrics_by_k(df, published_k=4, selector_k=2).to_dict()

        domain = set(spec["layer"][0]["encoding"]["color"]["scale"]["domain"])
        self.assertEqual(domain, {row["metric"] for row in _layer_data(spec, 0)})

    def test_every_measure_gets_its_own_colour(self) -> None:
        spec = metrics_by_k(mock_cluster_metrics_df(), published_k=4, selector_k=2).to_dict()
        scale = spec["layer"][0]["encoding"]["color"]["scale"]
        self.assertEqual(len(scale["range"]), len(scale["domain"]))
        self.assertEqual(len(set(scale["range"])), len(scale["range"]))


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


class TestMetricsByKMarkers(unittest.TestCase):
    """Which cut the chart presents as the answer.

    The selector's score peaks at the bottom of the tested range by
    construction, so a chart that marks only its peak invites reading that as
    the result. The published cut leads; the peak is shown beside it.
    """

    def test_leads_with_the_published_cut(self) -> None:
        spec = metrics_by_k(
            mock_cluster_metrics_df(), published_k=4, selector_k=2
        ).to_dict()
        title = spec["title"]["text"]
        self.assertIn("Published: 4", title)
        self.assertIn("peaked at 2", title)

    def test_marks_both_cuts(self) -> None:
        spec = metrics_by_k(
            mock_cluster_metrics_df(), published_k=4, selector_k=2
        ).to_dict()
        marks = _layer_data(spec, 1)
        self.assertEqual(
            {(m["num_clusters"], m["what"]) for m in marks},
            {(4, "published"), (2, "selector's peak")},
        )

    def test_marks_one_cut_when_they_agree(self) -> None:
        spec = metrics_by_k(
            mock_cluster_metrics_df(), published_k=2, selector_k=2
        ).to_dict()
        self.assertEqual(len(_layer_data(spec, 1)), 1)
        self.assertNotIn("peaked at", spec["title"]["text"])

    def test_works_without_a_selector_value(self) -> None:
        spec = metrics_by_k(mock_cluster_metrics_df(), published_k=4).to_dict()
        self.assertEqual(len(_layer_data(spec, 1)), 1)
