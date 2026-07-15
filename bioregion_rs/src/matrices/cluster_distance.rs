//! Port of `src/matrices/cluster_distance.py` (`ClusterDistanceMatrix.build`).

use std::collections::{BTreeSet, HashMap};

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::to_py;

/// Linear-interpolation percentile over an already-sorted slice (matches
/// numpy's default `interpolation="linear"`, which is what sklearn's
/// `RobustScaler` uses internally).
fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q / 100.0 * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }
}

/// Scale each column in place to match `sklearn.preprocessing.RobustScaler`
/// (median-center, divide by IQR; an IQR of 0 leaves the column unscaled, per
/// sklearn's documented behavior).
fn robust_scale_columns(matrix: &mut [Vec<f64>], num_cols: usize) {
    let num_rows = matrix.len();
    for col in 0..num_cols {
        let mut values: Vec<f64> = (0..num_rows).map(|row| matrix[row][col]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = percentile_linear(&values, 50.0);
        let q1 = percentile_linear(&values, 25.0);
        let q3 = percentile_linear(&values, 75.0);
        let iqr = q3 - q1;
        let scale = if iqr == 0.0 { 1.0 } else { iqr };
        for row in matrix.iter_mut() {
            row[col] = (row[col] - median) / scale;
        }
    }
}

/// Bray-Curtis distance: `sum(|u_i - v_i|) / sum(|u_i + v_i|)`, replacing a
/// NaN/infinite result (e.g. both vectors all-zero) with 1.0 (maximum
/// distance), matching `np.nan_to_num(Y, nan=1.0, posinf=1.0, neginf=1.0)`.
fn braycurtis(u: &[f64], v: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (a, b) in u.iter().zip(v) {
        num += (a - b).abs();
        den += (a + b).abs();
    }
    let d = num / den;
    if d.is_nan() || d.is_infinite() {
        1.0
    } else {
        d
    }
}

/// Build a cluster-by-cluster Bray-Curtis distance matrix (RobustScaler ->
/// `pdist(braycurtis)`) from per-cluster taxon averages, as a condensed
/// (upper-triangular, scipy `pdist`-order) distance vector plus the cluster
/// IDs in row order.
///
/// Mirrors `ClusterDistanceMatrix.build` / `pivot_taxon_counts_for_clusters`.
/// Row/column order need not match Python's `pivot()` order exactly — both
/// RobustScaler (per-column) and Bray-Curtis (order-invariant sums) give
/// identical distances under any consistent column permutation, and this
/// function returns `cluster_ids` alongside the condensed vector precisely so
/// callers can look distances up by cluster ID rather than raw position.
#[pyfunction]
pub fn build_cluster_distance_matrix(
    cluster_taxa_stats_df: PyDataFrame,
) -> PyResult<(Vec<f64>, Vec<u32>)> {
    let df: DataFrame = cluster_taxa_stats_df.into();

    let cluster_ca = df
        .column("cluster")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .clone();
    let taxon_ca = df
        .column("taxonId")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .clone();
    let average_ca = df
        .column("average")
        .map_err(to_py)?
        .as_materialized_series()
        .f64()
        .map_err(to_py)?
        .clone();

    let rows: Vec<(u32, u32, f64)> = cluster_ca
        .iter()
        .zip(taxon_ca.into_no_null_iter())
        .zip(average_ca.into_no_null_iter())
        .filter_map(|((cluster, taxon_id), average)| cluster.map(|c| (c, taxon_id, average)))
        .collect();

    let cluster_ids: Vec<u32> = rows
        .iter()
        .map(|&(c, _, _)| c)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let taxon_ids: Vec<u32> = rows
        .iter()
        .map(|&(_, t, _)| t)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    if cluster_ids.len() <= 1 {
        return Err(PyValueError::new_err(
            "More than one cluster is required to calculate distances",
        ));
    }

    let cluster_index: HashMap<u32, usize> = cluster_ids
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    let taxon_index: HashMap<u32, usize> =
        taxon_ids.iter().enumerate().map(|(i, &t)| (t, i)).collect();

    let mut matrix = vec![vec![0.0f64; taxon_ids.len()]; cluster_ids.len()];
    for (cluster, taxon_id, average) in rows {
        matrix[cluster_index[&cluster]][taxon_index[&taxon_id]] = average;
    }

    robust_scale_columns(&mut matrix, taxon_ids.len());

    let mut condensed = Vec::with_capacity(cluster_ids.len() * (cluster_ids.len() - 1) / 2);
    for i in 0..cluster_ids.len() {
        for j in (i + 1)..cluster_ids.len() {
            condensed.push(braycurtis(&matrix[i], &matrix[j]));
        }
    }

    Ok((condensed, cluster_ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy_linear() {
        let sorted = [1.0, 2.0, 3.0, 4.0];
        // numpy.percentile([1,2,3,4], 50) == 2.5
        assert_eq!(percentile_linear(&sorted, 50.0), 2.5);
        // numpy.percentile([1,2,3,4], 25) == 1.75
        assert_eq!(percentile_linear(&sorted, 25.0), 1.75);
    }

    #[test]
    fn braycurtis_matches_known_value() {
        // scipy.spatial.distance.braycurtis([1,0,0],[0,1,0]) == 1.0
        assert_eq!(braycurtis(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]), 1.0);
        // identical vectors -> 0 distance
        assert_eq!(braycurtis(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn braycurtis_all_zero_maps_to_max_distance() {
        assert_eq!(braycurtis(&[0.0, 0.0], &[0.0, 0.0]), 1.0);
    }
}
