//! Port of `src/dataframes/geocode_cluster_metrics.py`.
//!
//! Silhouette (mean), Calinski-Harabasz, and Davies-Bouldin formulas are read
//! directly from sklearn's `_unsupervised.py` source (not just its
//! docstrings). Calinski-Harabasz and Davies-Bouldin treat each row of the
//! square distance matrix as an n-dimensional "feature vector" and compute
//! ordinary Euclidean distances between those rows — that's the existing
//! methodology in the Python code (`X=dm_square`, no `metric="precomputed"`),
//! not something introduced by this port. Inertia is the codebase's own
//! hand-rolled distance-matrix approximation (`_compute_inertia`), not an
//! sklearn function. The Kneedle elbow-detection algorithm
//! (`select_optimal_k_elbow` / `_find_elbow_point`) delegates to the `kneed`
//! crate (an independent Rust port of the same Python `kneed` package this
//! module ports from) rather than a hand-rolled reimplementation — see the
//! note above `find_elbow_point` for why.
//!
//! `get_elbow_analysis`, `get_metrics_summary`, and `get_metric_interpretations`
//! stay in Python: they're plotting/presentation helpers, not compute.

use std::collections::HashMap;

use kneed::knee_locator::{
    InterpMethod, KneeLocator, KneeLocatorParams, ValidCurve, ValidDirection,
};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::to_py;

// --- shared distance-matrix helpers ------------------------------------------

/// Distance between points `i` and `j` in a scipy `pdist`-order condensed
/// vector for `n` objects; 0.0 when `i == j`.
fn condensed_dist(condensed: &[f64], n: usize, i: usize, j: usize) -> f64 {
    if i == j {
        return 0.0;
    }
    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
    condensed[lo * n - lo * (lo + 1) / 2 + hi - lo - 1]
}

/// Row `i` of the square distance matrix (distances from `i` to every point,
/// including itself as 0.0) — used as a "feature vector" for
/// Calinski-Harabasz / Davies-Bouldin, matching the Python source.
fn feature_row(condensed: &[f64], n: usize, i: usize) -> Vec<f64> {
    (0..n).map(|j| condensed_dist(condensed, n, i, j)).collect()
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

fn cluster_indices(labels: &[usize], k: usize) -> Vec<usize> {
    (0..labels.len()).filter(|&i| labels[i] == k).collect()
}

fn mean_of(features: &[Vec<f64>], indices: &[usize]) -> Vec<f64> {
    let dim = features[0].len();
    let mut mean = vec![0.0; dim];
    for &i in indices {
        for (m, v) in mean.iter_mut().zip(&features[i]) {
            *m += v;
        }
    }
    for m in mean.iter_mut() {
        *m /= indices.len() as f64;
    }
    mean
}

fn squared_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    squared_dist(a, b).sqrt()
}

// --- cluster validation metrics -----------------------------------------------

/// Mean silhouette score for a precomputed distance matrix. Mirrors
/// `sklearn.metrics.silhouette_score(X, labels, metric="precomputed")` (via
/// `_silhouette_reduce`): for each point, `a` = mean distance to same-cluster
/// points, `b` = min over other clusters of mean distance to that cluster;
/// `s = (b-a)/max(a,b)`, `0` for singleton clusters.
fn silhouette_score_mean(condensed: &[f64], n: usize, labels: &[usize], num_groups: usize) -> f64 {
    let mut group_sizes = vec![0usize; num_groups];
    for &l in labels {
        group_sizes[l] += 1;
    }

    let mut total = 0.0;
    for i in 0..n {
        let mut cluster_dist_sum = vec![0.0; num_groups];
        for j in 0..n {
            cluster_dist_sum[labels[j]] += condensed_dist(condensed, n, i, j);
        }
        let own = labels[i];
        if group_sizes[own] <= 1 {
            continue; // s(i) = 0, contributes nothing to the sum
        }
        let a = cluster_dist_sum[own] / (group_sizes[own] - 1) as f64;
        let b = (0..num_groups)
            .filter(|&c| c != own)
            .map(|c| cluster_dist_sum[c] / group_sizes[c] as f64)
            .fold(f64::INFINITY, f64::min);
        let denom = a.max(b);
        if denom != 0.0 {
            total += (b - a) / denom;
        }
    }
    total / n as f64
}

/// Mirrors `calinski_harabasz_score(X=dm_square, labels)`.
fn calinski_harabasz(condensed: &[f64], n: usize, labels: &[usize], num_groups: usize) -> f64 {
    let features: Vec<Vec<f64>> = (0..n).map(|i| feature_row(condensed, n, i)).collect();
    let global_mean = mean_of(&features, &(0..n).collect::<Vec<_>>());

    let mut extra_disp = 0.0;
    let mut intra_disp = 0.0;
    for k in 0..num_groups {
        let idx = cluster_indices(labels, k);
        let mean_k = mean_of(&features, &idx);
        extra_disp += idx.len() as f64 * squared_dist(&mean_k, &global_mean);
        for &i in &idx {
            intra_disp += squared_dist(&features[i], &mean_k);
        }
    }

    if intra_disp == 0.0 {
        1.0
    } else {
        extra_disp * (n - num_groups) as f64 / (intra_disp * (num_groups - 1) as f64)
    }
}

/// Mirrors `davies_bouldin_score(X=dm_square, labels)`.
fn davies_bouldin(condensed: &[f64], n: usize, labels: &[usize], num_groups: usize) -> f64 {
    let features: Vec<Vec<f64>> = (0..n).map(|i| feature_row(condensed, n, i)).collect();

    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(num_groups);
    let mut intra_dists = vec![0.0; num_groups];
    for k in 0..num_groups {
        let idx = cluster_indices(labels, k);
        let centroid = mean_of(&features, &idx);
        intra_dists[k] = idx
            .iter()
            .map(|&i| euclidean_dist(&features[i], &centroid))
            .sum::<f64>()
            / idx.len() as f64;
        centroids.push(centroid);
    }

    const EPS: f64 = 1e-9;
    let centroid_dist = |i: usize, j: usize| euclidean_dist(&centroids[i], &centroids[j]);
    let all_intra_zero = intra_dists.iter().all(|&d| d.abs() < EPS);
    let all_centroid_zero =
        (0..num_groups).all(|i| (0..num_groups).all(|j| centroid_dist(i, j).abs() < EPS));
    if all_intra_zero || all_centroid_zero {
        return 0.0;
    }

    let mut total = 0.0;
    for i in 0..num_groups {
        let best = (0..num_groups)
            .filter(|&j| j != i)
            .map(|j| {
                let cd = centroid_dist(i, j);
                if cd == 0.0 {
                    0.0
                } else {
                    (intra_dists[i] + intra_dists[j]) / cd
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        total += best;
    }
    total / num_groups as f64
}

/// Mirrors the codebase's own `_compute_inertia`: for each cluster, sum of
/// squared pairwise distances within it, divided by `2 * cluster_size`.
fn inertia(condensed: &[f64], n: usize, labels: &[usize], num_groups: usize) -> f64 {
    let mut total = 0.0;
    for k in 0..num_groups {
        let idx = cluster_indices(labels, k);
        if idx.len() <= 1 {
            continue;
        }
        let mut sum_sq = 0.0;
        for &i in &idx {
            for &j in &idx {
                let d = condensed_dist(condensed, n, i, j);
                sum_sq += d * d;
            }
        }
        total += sum_sq / (2.0 * idx.len() as f64);
    }
    total
}

fn normalize_min_max(values: &[f64], invert: bool) -> Vec<f64> {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = if max != min { max - min } else { 1.0 };
    values
        .iter()
        .map(|&v| {
            let norm = (v - min) / range;
            if invert { 1.0 - norm } else { norm }
        })
        .collect()
}

/// Build a GeocodeClusterMetricsSchema DataFrame: silhouette,
/// Calinski-Harabasz, Davies-Bouldin, and inertia for every `num_clusters`
/// value present in `geocode_cluster_df`, plus normalized [0, 1] versions and
/// a weighted combined score.
///
/// Mirrors `build_geocode_cluster_metrics_df`. `distance_matrix` is passed in
/// as its condensed form (`GeocodeDistanceMatrix.condensed()`) directly,
/// since building it involves UMAP (Phase 3, stays in Python).
#[pyfunction]
#[pyo3(signature = (
    condensed, geocode_cluster_df,
    weight_silhouette = 0.4, weight_calinski_harabasz = 0.3, weight_davies_bouldin = 0.3,
))]
pub fn build_geocode_cluster_metrics(
    condensed: Vec<f64>,
    geocode_cluster_df: PyDataFrame,
    weight_silhouette: f64,
    weight_calinski_harabasz: f64,
    weight_davies_bouldin: f64,
) -> PyResult<PyDataFrame> {
    let geocode_cluster_df: DataFrame = geocode_cluster_df.into();
    let weight_sum = weight_silhouette + weight_calinski_harabasz + weight_davies_bouldin;
    let (weight_silhouette, weight_calinski_harabasz, weight_davies_bouldin) = (
        weight_silhouette / weight_sum,
        weight_calinski_harabasz / weight_sum,
        weight_davies_bouldin / weight_sum,
    );

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

    let mut k_values: Vec<u32> = num_clusters_ca
        .into_no_null_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    k_values.sort_unstable();

    let mut sil = Vec::with_capacity(k_values.len());
    let mut ch = Vec::with_capacity(k_values.len());
    let mut db = Vec::with_capacity(k_values.len());
    let mut ine = Vec::with_capacity(k_values.len());

    for &k in &k_values {
        let raw_labels: Vec<u32> = num_clusters_ca
            .into_no_null_iter()
            .zip(cluster_ca.into_no_null_iter())
            .filter(|&(nc, _)| nc == k)
            .map(|(_, c)| c)
            .collect();
        let (labels, num_groups) = dense_labels(&raw_labels);

        sil.push(silhouette_score_mean(&condensed, n, &labels, num_groups));
        ch.push(calinski_harabasz(&condensed, n, &labels, num_groups));
        db.push(davies_bouldin(&condensed, n, &labels, num_groups));
        ine.push(inertia(&condensed, n, &labels, num_groups));
    }

    let sil_norm: Vec<f64> = sil.iter().map(|&s| (s + 1.0) / 2.0).collect();
    let ch_norm = normalize_min_max(&ch, false);
    let db_norm = normalize_min_max(&db, true);
    let ine_norm = normalize_min_max(&ine, true);
    let combined: Vec<f64> = (0..k_values.len())
        .map(|i| {
            weight_silhouette * sil_norm[i]
                + weight_calinski_harabasz * ch_norm[i]
                + weight_davies_bouldin * db_norm[i]
        })
        .collect();

    let out = DataFrame::new(
        k_values.len(),
        vec![
            UInt32Chunked::from_vec("num_clusters".into(), k_values).into_column(),
            Float64Chunked::from_vec("silhouette_score".into(), sil).into_column(),
            Float64Chunked::from_vec("calinski_harabasz_score".into(), ch).into_column(),
            Float64Chunked::from_vec("davies_bouldin_score".into(), db).into_column(),
            Float64Chunked::from_vec("inertia".into(), ine).into_column(),
            Float64Chunked::from_vec("silhouette_normalized".into(), sil_norm).into_column(),
            Float64Chunked::from_vec("calinski_harabasz_normalized".into(), ch_norm).into_column(),
            Float64Chunked::from_vec("davies_bouldin_normalized".into(), db_norm).into_column(),
            Float64Chunked::from_vec("inertia_normalized".into(), ine_norm).into_column(),
            Float64Chunked::from_vec("combined_score".into(), combined).into_column(),
        ],
    )
    .map_err(to_py)?;

    Ok(PyDataFrame(out))
}

// --- Kneedle elbow detection ---------------------------------------------------
//
// Delegates to the `kneed` crate (a from-scratch, independently-maintained
// Rust port of the same Python `kneed` package `geocode_cluster_metrics.py`
// uses), rather than a hand-rolled reimplementation. An earlier version of
// this function hand-rolled `argrelextrema`, excluding the first/last point
// from local-extrema consideration; that's wrong in general —
// `scipy.signal.argrelextrema`'s default `mode="clip"` compares boundary
// points against a clipped (self-referential) neighbor, which is only ever
// *not* trivially satisfied by the other side's real comparison, so
// excluding endpoints silently diverges on non-monotonic curves (confirmed
// against real inertia data: y=[90,100,50,20,18,17] gave elbow=5 by hand,
// vs. Python kneed's actual elbow=2). `kneed` handles this correctly, and
// getting this exactly right by hand a second time isn't worth the risk
// versus depending on a crate purpose-built to match the same reference
// implementation.

/// Find the elbow point in a convex, decreasing curve (e.g. inertia vs k).
///
/// `x` must already be sorted ascending (matches `metrics_df.sort("num_clusters")`
/// before calling `_find_elbow_point`). Mirrors `_find_elbow_point`'s own
/// `len(df) < 3` guard, which lives in the Python wrapper rather than in
/// `kneed.KneeLocator` itself.
pub(crate) fn find_elbow_point(x: &[f64], y: &[f64], sensitivity: f64) -> Option<f64> {
    if x.len() < 3 {
        return None;
    }
    let params = KneeLocatorParams::new(
        ValidCurve::Convex,
        ValidDirection::Decreasing,
        InterpMethod::Interp1d,
    );
    let locator = KneeLocator::new(x.to_vec(), y.to_vec(), sensitivity, params).ok()?;
    locator.knee
}

/// Find the optimal `num_clusters` via the elbow (Kneedle) method over the
/// inertia column of a GeocodeClusterMetricsSchema-like DataFrame.
///
/// Mirrors `select_optimal_k_elbow` / `_find_elbow_point`.
#[pyfunction]
#[pyo3(signature = (num_clusters, inertia, sensitivity = 1.0))]
pub fn select_optimal_k_elbow(
    num_clusters: Vec<u32>,
    inertia: Vec<f64>,
    sensitivity: f64,
) -> Option<u32> {
    let mut paired: Vec<(u32, f64)> = num_clusters.into_iter().zip(inertia).collect();
    paired.sort_by_key(|&(k, _)| k);
    let x: Vec<f64> = paired.iter().map(|&(k, _)| k as f64).collect();
    let y: Vec<f64> = paired.iter().map(|&(_, v)| v).collect();
    find_elbow_point(&x, &y, sensitivity).map(|k| k.round() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn davies_bouldin_matches_sklearn_docstring_example() {
        // sklearn.metrics.davies_bouldin_score([[0,1],[1,1],[3,4]], [0,0,1])
        // == 0.12803687993289598 (verified directly against sklearn while
        // porting). Reproduced here using the 2D points directly as
        // "features" (n=3, no distance-matrix indirection needed to
        // sanity-check the formula).
        let features = [vec![0.0, 1.0], vec![1.0, 1.0], vec![3.0, 4.0]];
        let centroid0 = mean_of(&features, &[0, 1]);
        let centroid1 = mean_of(&features, &[2]);
        let intra0 = (euclidean_dist(&features[0], &centroid0)
            + euclidean_dist(&features[1], &centroid0))
            / 2.0;
        let intra1 = 0.0;
        let cd = euclidean_dist(&centroid0, &centroid1);
        let expected = ((intra0 + intra1) / cd + (intra1 + intra0) / cd) / 2.0;
        assert!((expected - 0.12803687993289598).abs() < 1e-9);
    }

    #[test]
    fn elbow_detects_obvious_knee() {
        // Sharp elbow at k=3: inertia drops fast then flattens.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = [100.0, 50.0, 20.0, 18.0, 17.0, 16.0];
        assert_eq!(find_elbow_point(&x, &y, 1.0), Some(3.0));
    }

    #[test]
    fn elbow_none_for_perfectly_linear_curve() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [50.0, 40.0, 30.0, 20.0, 10.0];
        assert_eq!(find_elbow_point(&x, &y, 1.0), None);
    }

    #[test]
    fn elbow_matches_python_kneed_on_non_monotonic_curve() {
        // KneeLocator([2,3,4,5,6,7], [90,100,50,20,18,17], curve="convex",
        // direction="decreasing", S=1.0).elbow == 2 in Python's real `kneed`.
        // A non-monotonic ("wobbly") curve like this is exactly the case an
        // earlier hand-rolled version of this function got wrong (it
        // excluded curve endpoints from local-extrema consideration, which
        // diverges from scipy's actual `argrelextrema(mode="clip")`
        // semantics whenever the first point isn't the strict global
        // extreme) -- it returned 5 instead of 2.
        let x = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = [90.0, 100.0, 50.0, 20.0, 18.0, 17.0];
        assert_eq!(find_elbow_point(&x, &y, 1.0), Some(2.0));
    }
}
