import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import dataframely as dy
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.matrices.geocode_distance import GeocodeDistanceMatrix

logger = logging.getLogger(__name__)


@dataclass
class ClusterMetrics:
    """Metrics for evaluating a specific cluster count"""

    k: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    inertia: float
    # Average cluster size
    mean_cluster_size: int
    # Size of smallest cluster
    min_cluster_size: int
    # Size of largest cluster
    max_cluster_size: int


@dataclass
class OptimalKResult:
    """Result of optimal k selection"""

    optimal_k: int
    all_metrics: List[ClusterMetrics]
    selection_method: str
    reason: str


class ClusterOptimizer:
    """
    Evaluates different numbers of clusters to find the optimal k.

    Uses multiple clustering quality metrics:
    - Silhouette Score: Measures how similar objects are to their own cluster
      compared to other clusters. Range [-1, 1], higher is better.
    - Davies-Bouldin Index: Ratio of within-cluster to between-cluster distances.
      Lower is better.
    - Calinski-Harabasz Score: Ratio of between-cluster to within-cluster variance.
      Higher is better.
    - Inertia: Sum of squared distances to nearest cluster center.
      Lower is better, but look for "elbow" in curve.
    """

    def __init__(
        self,
        geocode_dataframe: dy.DataFrame[GeocodeNoEdgesSchema],
        distance_matrix: GeocodeDistanceMatrix,
        connectivity_matrix: GeocodeConnectivityMatrix,
    ):
        self.geocode_dataframe = geocode_dataframe
        self.distance_matrix = distance_matrix
        self.connectivity_matrix = connectivity_matrix
        self._square_distances = distance_matrix.squareform()

    def evaluate_k_range(
        self,
        k_min: int = 5,
        k_max: int = 20,
    ) -> List[ClusterMetrics]:
        """
        Evaluate clustering quality for a range of k values.

        Args:
            k_min: Minimum number of clusters to evaluate
            k_max: Maximum number of clusters to evaluate

        Returns:
            List of ClusterMetrics for each k value
        """
        n_geocodes = len(self.geocode_dataframe)

        if k_max >= n_geocodes:
            logger.warning(
                f"k_max ({k_max}) >= number of geocodes ({n_geocodes}). "
                f"Reducing k_max to {n_geocodes - 1}"
            )
            k_max = n_geocodes - 1

        if k_min < 2:
            logger.warning(f"k_min must be at least 2. Setting k_min=2")
            k_min = 2

        metrics_list: List[ClusterMetrics] = []

        logger.info(f"Evaluating cluster counts from {k_min} to {k_max}")

        for k in range(k_min, k_max + 1):
            logger.info(f"Evaluating k={k}...")

            # Perform clustering
            clustering = AgglomerativeClustering(
                n_clusters=k,
                connectivity=csr_matrix(self.connectivity_matrix._connectivity_matrix),
                linkage="ward",
            )
            labels = clustering.fit_predict(self._square_distances)

            # Calculate metrics
            metrics = self._calculate_metrics(k, labels)
            metrics_list.append(metrics)

            logger.info(
                f"k={k}: silhouette={metrics.silhouette:.3f}, "
                f"davies_bouldin={metrics.davies_bouldin:.3f}, "
                f"calinski_harabasz={metrics.calinski_harabasz:.1f}"
            )

        return metrics_list

    def _calculate_metrics(self, k: int, labels: np.ndarray) -> ClusterMetrics:
        """Calculate all clustering quality metrics for a given clustering"""

        # Silhouette score (higher is better, range [-1, 1])
        silhouette = silhouette_score(
            self._square_distances, labels, metric="precomputed"
        )

        # Davies-Bouldin index (lower is better)
        davies_bouldin = davies_bouldin_score(self._square_distances, labels)

        # Calinski-Harabasz score (higher is better)
        # This requires the original feature matrix, not distances
        # We'll use the distance matrix as a proxy
        calinski_harabasz = calinski_harabasz_score(self._square_distances, labels)

        # Calculate inertia (sum of squared distances to cluster centers)
        inertia = self._calculate_inertia(labels)

        # Cluster size statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        mean_cluster_size = int(np.mean(counts))
        min_cluster_size = int(np.min(counts))
        max_cluster_size = int(np.max(counts))

        return ClusterMetrics(
            k=k,
            silhouette=float(silhouette),
            davies_bouldin=float(davies_bouldin),
            calinski_harabasz=float(calinski_harabasz),
            inertia=float(inertia),
            mean_cluster_size=mean_cluster_size,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )

    def _calculate_inertia(self, labels: np.ndarray) -> float:
        """Calculate inertia (sum of squared distances to cluster centers)"""
        inertia = 0.0

        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_distances = self._square_distances[cluster_mask][:, cluster_mask]

            # Use centroid as cluster center
            # For distance matrix, centroid is the point with minimum sum of distances
            sum_distances = cluster_distances.sum(axis=1)
            centroid_idx = np.argmin(sum_distances)

            # Sum squared distances to centroid
            inertia += np.sum(cluster_distances[centroid_idx] ** 2)

        return inertia

    def suggest_optimal_k(
        self, metrics_list: List[ClusterMetrics], method: str = "multi_criteria"
    ) -> OptimalKResult:
        """
        Suggest optimal k based on evaluation metrics.

        Args:
            metrics_list: List of ClusterMetrics from evaluate_k_range()
            method: Selection method. Options:
                - "multi_criteria": Weighted combination of all metrics (default)
                - "silhouette": Choose k with highest silhouette score
                - "elbow": Find elbow in inertia curve
                - "compromise": Balance between silhouette and cluster count

        Returns:
            OptimalKResult with suggested k and reasoning
        """
        if method == "silhouette":
            return self._suggest_by_silhouette(metrics_list)
        elif method == "elbow":
            return self._suggest_by_elbow(metrics_list)
        elif method == "compromise":
            return self._suggest_by_compromise(metrics_list)
        else:  # multi_criteria
            return self._suggest_by_multi_criteria(metrics_list)

    def _suggest_by_silhouette(
        self, metrics_list: List[ClusterMetrics]
    ) -> OptimalKResult:
        """Choose k with highest silhouette score"""
        best_metrics = max(metrics_list, key=lambda m: m.silhouette)
        return OptimalKResult(
            optimal_k=best_metrics.k,
            all_metrics=metrics_list,
            selection_method="silhouette",
            reason=f"Highest silhouette score: {best_metrics.silhouette:.3f}",
        )

    def _suggest_by_elbow(self, metrics_list: List[ClusterMetrics]) -> OptimalKResult:
        """Find elbow point in inertia curve using derivative method"""
        k_values = [m.k for m in metrics_list]
        inertias = [m.inertia for m in metrics_list]

        # Calculate second derivative to find elbow
        first_deriv = np.diff(inertias)
        second_deriv = np.diff(first_deriv)

        # Elbow is where second derivative is maximum (most negative change)
        elbow_idx = np.argmax(second_deriv) + 1  # +1 because of double diff
        optimal_k = k_values[elbow_idx]

        return OptimalKResult(
            optimal_k=optimal_k,
            all_metrics=metrics_list,
            selection_method="elbow",
            reason=f"Elbow point in inertia curve at k={optimal_k}",
        )

    def _suggest_by_compromise(
        self, metrics_list: List[ClusterMetrics]
    ) -> OptimalKResult:
        """
        Balance between good silhouette score and moderate cluster count.
        Prefers fewer clusters if silhouette scores are similar.
        """
        # Find k values where silhouette > 0.3 (reasonable quality)
        good_quality = [m for m in metrics_list if m.silhouette > 0.3]

        if not good_quality:
            # No good quality found, fall back to best silhouette
            # but keep the compromise method name
            best_metrics = max(metrics_list, key=lambda m: m.silhouette)
            return OptimalKResult(
                optimal_k=best_metrics.k,
                all_metrics=metrics_list,
                selection_method="compromise",
                reason=f"No clusters with silhouette > 0.3 found. Using highest silhouette score: {best_metrics.silhouette:.3f}",
            )

        # Among good quality, prefer lower k (simpler solution)
        best_metrics = min(good_quality, key=lambda m: m.k)

        return OptimalKResult(
            optimal_k=best_metrics.k,
            all_metrics=metrics_list,
            selection_method="compromise",
            reason=f"Lowest k with silhouette > 0.3: k={best_metrics.k} (score={best_metrics.silhouette:.3f})",
        )

    def _suggest_by_multi_criteria(
        self, metrics_list: List[ClusterMetrics]
    ) -> OptimalKResult:
        """
        Use weighted ranking across multiple metrics.

        Ranks each k by each metric and finds k with best average rank.
        This balances all quality indicators.
        """
        k_values = [m.k for m in metrics_list]

        # Rank by silhouette (higher is better)
        silhouette_ranks = self._rank_ascending([m.silhouette for m in metrics_list])

        # Rank by davies_bouldin (lower is better, so invert)
        db_ranks = self._rank_descending([m.davies_bouldin for m in metrics_list])

        # Rank by calinski_harabasz (higher is better)
        ch_ranks = self._rank_ascending([m.calinski_harabasz for m in metrics_list])

        # Check for balanced cluster sizes (penalize very unbalanced)
        balance_scores = []
        for m in metrics_list:
            # Ratio of min to mean cluster size (1.0 = perfect balance)
            balance = m.min_cluster_size / m.mean_cluster_size
            balance_scores.append(balance)
        balance_ranks = self._rank_ascending(balance_scores)

        # Calculate weighted average rank
        # Silhouette is most important, then Davies-Bouldin, then others
        weights = {
            "silhouette": 0.4,
            "davies_bouldin": 0.3,
            "calinski_harabasz": 0.2,
            "balance": 0.1,
        }

        avg_ranks = []
        for i in range(len(k_values)):
            avg_rank = (
                weights["silhouette"] * silhouette_ranks[i]
                + weights["davies_bouldin"] * db_ranks[i]
                + weights["calinski_harabasz"] * ch_ranks[i]
                + weights["balance"] * balance_ranks[i]
            )
            avg_ranks.append(avg_rank)

        # Find k with best (lowest) average rank
        best_idx = np.argmin(avg_ranks)
        optimal_k = k_values[best_idx]
        best_metrics = metrics_list[best_idx]

        return OptimalKResult(
            optimal_k=optimal_k,
            all_metrics=metrics_list,
            selection_method="multi_criteria",
            reason=(
                f"Best average rank across metrics: "
                f"silhouette={best_metrics.silhouette:.3f}, "
                f"DB={best_metrics.davies_bouldin:.3f}, "
                f"CH={best_metrics.calinski_harabasz:.1f}"
            ),
        )

    @staticmethod
    def _rank_ascending(values: List[float]) -> List[float]:
        """Rank values where higher is better (returns normalized ranks 0-1)"""
        sorted_indices = np.argsort(values)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(len(values))
        # Normalize to 0-1
        return (ranks / (len(values) - 1)).tolist()

    @staticmethod
    def _rank_descending(values: List[float]) -> List[float]:
        """Rank values where lower is better (returns normalized ranks 0-1)"""
        sorted_indices = np.argsort(values)[::-1]  # Reverse for descending
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(len(values))
        # Normalize to 0-1
        return (ranks / (len(values) - 1)).tolist()


def metrics_to_dataframe(metrics_list: List[ClusterMetrics]) -> pl.DataFrame:
    """Convert list of ClusterMetrics to a Polars DataFrame for easy analysis"""
    return pl.DataFrame(
        {
            "k": [m.k for m in metrics_list],
            "silhouette": [m.silhouette for m in metrics_list],
            "davies_bouldin": [m.davies_bouldin for m in metrics_list],
            "calinski_harabasz": [m.calinski_harabasz for m in metrics_list],
            "inertia": [m.inertia for m in metrics_list],
            "mean_cluster_size": [m.mean_cluster_size for m in metrics_list],
            "min_cluster_size": [m.min_cluster_size for m in metrics_list],
            "max_cluster_size": [m.max_cluster_size for m in metrics_list],
        }
    )


def create_metrics_report(result: OptimalKResult) -> str:
    """
    Create a human-readable report of cluster optimization results.

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CLUSTER OPTIMIZATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Selection Method: {result.selection_method}")
    lines.append(f"Optimal k: {result.optimal_k}")
    lines.append(f"Reason: {result.reason}")
    lines.append("")
    lines.append("All evaluated cluster counts:")
    lines.append("-" * 70)
    lines.append(
        f"{'k':<4} {'Silhouette':<12} {'Davies-Bouldin':<16} {'Calinski-H':<12} {'Min Size':<10}"
    )
    lines.append("-" * 70)

    for metrics in result.all_metrics:
        marker = "→" if metrics.k == result.optimal_k else " "
        lines.append(
            f"{marker} {metrics.k:<3} {metrics.silhouette:<12.3f} "
            f"{metrics.davies_bouldin:<16.3f} {metrics.calinski_harabasz:<12.1f} "
            f"{metrics.min_cluster_size:<10}"
        )

    lines.append("-" * 70)
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  • Silhouette: Higher is better (range: -1 to 1)")
    lines.append("  • Davies-Bouldin: Lower is better (0 = perfect)")
    lines.append("  • Calinski-Harabasz: Higher is better")
    lines.append("  • Min Size: Minimum number of hexagons in smallest cluster")
    lines.append("")

    return "\n".join(lines)
