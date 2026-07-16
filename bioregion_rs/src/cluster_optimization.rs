//! Port of `src/cluster_optimization.py` (`optimize_num_clusters`).
//!
//! Thin orchestration over already-ported pieces: build the full cluster
//! validation metrics table (`geocode_cluster_metrics::build_geocode_cluster_metrics`),
//! then pick `k` via the elbow method
//! (`geocode_cluster_metrics::find_elbow_point`), falling back to the `k`
//! with the highest `combined_score` if no elbow point is found.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::dataframes::geocode_cluster_metrics::{build_geocode_cluster_metrics, find_elbow_point};
use crate::to_py;

/// Find the optimal number of clusters via the elbow method, computing
/// cluster validation metrics for every `num_clusters` value along the way.
///
/// Mirrors `optimize_num_clusters`. Returns `(optimal_k, metrics_df)`.
#[pyfunction]
#[pyo3(signature = (
    condensed, geocode_cluster_df, elbow_sensitivity = 1.0,
    weight_silhouette = 0.4, weight_calinski_harabasz = 0.3, weight_davies_bouldin = 0.3,
))]
#[allow(clippy::too_many_arguments)]
pub fn optimize_num_clusters(
    condensed: Vec<f64>,
    geocode_cluster_df: PyDataFrame,
    elbow_sensitivity: f64,
    weight_silhouette: f64,
    weight_calinski_harabasz: f64,
    weight_davies_bouldin: f64,
) -> PyResult<(u32, PyDataFrame)> {
    let metrics_df: DataFrame = build_geocode_cluster_metrics(
        condensed,
        geocode_cluster_df,
        weight_silhouette,
        weight_calinski_harabasz,
        weight_davies_bouldin,
    )?
    .0;

    // build_geocode_cluster_metrics already emits rows in ascending
    // num_clusters order, matching find_elbow_point's "x sorted ascending"
    // requirement.
    let num_clusters_ca = metrics_df
        .column("num_clusters")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .clone();
    let inertia_ca = metrics_df
        .column("inertia")
        .map_err(to_py)?
        .as_materialized_series()
        .f64()
        .map_err(to_py)?
        .clone();
    let k_values: Vec<f64> = num_clusters_ca.into_no_null_iter().map(f64::from).collect();
    let inertia_values: Vec<f64> = inertia_ca.into_no_null_iter().collect();

    let optimal_k: u32 = match find_elbow_point(&k_values, &inertia_values, elbow_sensitivity) {
        Some(k) => k.round() as u32,
        None => {
            // Fallback: k with the highest combined_score.
            let combined_ca = metrics_df
                .column("combined_score")
                .map_err(to_py)?
                .as_materialized_series()
                .f64()
                .map_err(to_py)?
                .clone();
            let best_idx = combined_ca
                .into_no_null_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .expect("metrics_df has at least one row");
            num_clusters_ca
                .get(best_idx)
                .expect("combined_score and num_clusters columns have equal length")
        }
    };

    Ok((optimal_k, PyDataFrame(metrics_df)))
}
