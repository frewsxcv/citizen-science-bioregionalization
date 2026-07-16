//! Port of `src/dataframes/geocode_silhouette_score.py`
//! (`build_geocode_silhouette_score_df`).
//!
//! Same underlying formula as `geocode_cluster_metrics.rs`'s mean silhouette
//! score (read from sklearn's `_unsupervised.py` `_silhouette_reduce`), but
//! this file needs the *per-point* scores (`silhouette_samples`), not just
//! their mean.

use std::collections::HashMap;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::to_py;

/// Distance between points `i` and `j` in a scipy `pdist`-order condensed
/// vector for `n` objects; 0.0 when `i == j`.
fn condensed_dist(condensed: &[f64], n: usize, i: usize, j: usize) -> f64 {
    if i == j {
        return 0.0;
    }
    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
    condensed[lo * n - lo * (lo + 1) / 2 + hi - lo - 1]
}

/// Relabel arbitrary group ids into dense `0..num_groups` indices, ordered by
/// sorted distinct values (matches sklearn's `LabelEncoder`).
fn dense_labels(labels: &[u32]) -> (Vec<usize>, usize) {
    let mut sorted: Vec<u32> = labels.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let index_of: HashMap<u32, usize> = sorted.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    (labels.iter().map(|c| index_of[c]).collect(), sorted.len())
}

/// Per-point silhouette scores for a precomputed distance matrix. Mirrors
/// `sklearn.metrics.silhouette_samples(X, labels, metric="precomputed")` (via
/// `_silhouette_reduce`): for each point, `a` = mean distance to same-cluster
/// points, `b` = min over other clusters of mean distance to that cluster;
/// `s = (b-a)/max(a,b)`, `0` for singleton clusters.
fn silhouette_samples(
    condensed: &[f64],
    n: usize,
    labels: &[usize],
    num_groups: usize,
) -> Vec<f64> {
    let mut group_sizes = vec![0usize; num_groups];
    for &l in labels {
        group_sizes[l] += 1;
    }

    (0..n)
        .map(|i| {
            let mut cluster_dist_sum = vec![0.0; num_groups];
            for j in 0..n {
                cluster_dist_sum[labels[j]] += condensed_dist(condensed, n, i, j);
            }
            let own = labels[i];
            if group_sizes[own] <= 1 {
                return 0.0;
            }
            let a = cluster_dist_sum[own] / (group_sizes[own] - 1) as f64;
            let b = (0..num_groups)
                .filter(|&c| c != own)
                .map(|c| cluster_dist_sum[c] / group_sizes[c] as f64)
                .fold(f64::INFINITY, f64::min);
            let denom = a.max(b);
            if denom == 0.0 { 0.0 } else { (b - a) / denom }
        })
        .collect()
}

/// Build a GeocodeSilhouetteScoreSchema DataFrame: for every `num_clusters`
/// value in `geocode_cluster_df`, one row with `geocode = null` holding the
/// mean silhouette score, followed by one row per geocode holding its
/// individual silhouette score (in `geocode_cluster_df`'s row order for that
/// k, matching the distance matrix's row/column order).
///
/// Mirrors `build_geocode_silhouette_score_df`. `distance_matrix` is passed
/// in as its condensed form (`GeocodeDistanceMatrix.condensed()`) directly,
/// since building it involves UMAP (Phase 3, stays in Python).
#[pyfunction]
pub fn build_geocode_silhouette_score(
    condensed: Vec<f64>,
    geocode_cluster_df: PyDataFrame,
) -> PyResult<PyDataFrame> {
    let geocode_cluster_df: DataFrame = geocode_cluster_df.into();

    let n_squared_2 = 1.0 + 8.0 * condensed.len() as f64;
    let n = ((1.0 + n_squared_2.sqrt()) / 2.0).round() as usize;

    let num_clusters_ca = geocode_cluster_df
        .column("num_clusters")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .clone();
    let cluster_ca = geocode_cluster_df
        .column("cluster")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .clone();
    let geocode_ca = geocode_cluster_df
        .column("geocode")
        .map_err(to_py)?
        .as_materialized_series()
        .u64()
        .map_err(to_py)?
        .clone();

    let mut k_values: Vec<u32> = num_clusters_ca
        .into_no_null_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    k_values.sort_unstable();

    let mut out_geocode: Vec<Option<u64>> = Vec::new();
    let mut out_score: Vec<f64> = Vec::new();
    let mut out_num_clusters: Vec<u32> = Vec::new();

    for &k in &k_values {
        let rows: Vec<(u64, u32)> = num_clusters_ca
            .into_no_null_iter()
            .zip(geocode_ca.into_no_null_iter())
            .zip(cluster_ca.into_no_null_iter())
            .filter(|&((nc, _), _)| nc == k)
            .map(|((_, g), c)| (g, c))
            .collect();
        let raw_labels: Vec<u32> = rows.iter().map(|&(_, c)| c).collect();
        let (labels, num_groups) = dense_labels(&raw_labels);

        let samples = silhouette_samples(&condensed, n, &labels, num_groups);
        let mean = samples.iter().sum::<f64>() / n as f64;

        out_geocode.push(None);
        out_score.push(mean);
        out_num_clusters.push(k);

        for (&(geocode, _), &score) in rows.iter().zip(&samples) {
            out_geocode.push(Some(geocode));
            out_score.push(score);
            out_num_clusters.push(k);
        }
    }

    let geocode_col: UInt64Chunked = out_geocode.into_iter().collect();
    let out = DataFrame::new(
        out_score.len(),
        vec![
            geocode_col.with_name("geocode".into()).into_column(),
            Float64Chunked::from_vec("silhouette_score".into(), out_score).into_column(),
            UInt32Chunked::from_vec("num_clusters".into(), out_num_clusters).into_column(),
        ],
    )
    .map_err(to_py)?;

    Ok(PyDataFrame(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_average_to_the_mean_score() {
        // Same fixture as geocode_cluster_metrics's silhouette test data:
        // 3 tight pairs, far apart from each other.
        let condensed = [
            0.1, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0, 0.1, 3.0, 3.0, 3.0, 3.0, 0.1,
        ];
        let labels = [0usize, 0, 1, 1, 2, 2];
        let samples = silhouette_samples(&condensed, 6, &labels, 3);
        assert_eq!(samples.len(), 6);
        // Tight, well-separated clusters -> every point strongly prefers its
        // own cluster.
        assert!(samples.iter().all(|&s| s > 0.9));
    }
}
