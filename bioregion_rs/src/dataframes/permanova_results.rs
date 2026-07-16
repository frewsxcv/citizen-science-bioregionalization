//! Port of `src/dataframes/permanova_results.py` (`build_permanova_results_df`).
//!
//! PERMANOVA's pseudo-F statistic is a deterministic function of the distance
//! matrix and group assignment, and is ported exactly (verified bit-for-bit
//! against `skbio.stats.distance.permanova` with `permutations=0`). Its
//! p-value, however, comes from a Monte Carlo permutation test that
//! `skbio.stats.distance.permanova` runs with an *unseeded* random generator
//! (`seed=None` in `build_permanova_results_df`) — so the p-value isn't
//! bit-reproducible even between two separate Python runs, let alone between
//! Python and Rust. This port uses the same algorithm (shuffle the group
//! labels, recompute the statistic, `p = (1 + count(F_perm >= F_obs)) /
//! (1 + permutations)`) with Rust's own RNG, which is the best "equivalence"
//! achievable for an inherently randomized test.

use std::collections::HashMap;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rand::rng;
use rand::seq::SliceRandom;

use crate::to_py;

/// Distance between rows `i` and `j` (i != j) in a scipy `pdist`-order
/// condensed distance vector for `n` objects.
fn condensed_dist(condensed: &[f64], n: usize, i: usize, j: usize) -> f64 {
    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
    condensed[lo * n - lo * (lo + 1) / 2 + hi - lo - 1]
}

/// PERMANOVA pseudo-F statistic for a given permutation of group labels.
/// Mirrors `permanova_f_stat_sW_cy` + `_compute_f_stat`.
fn permanova_f_stat(
    condensed: &[f64],
    n: usize,
    grouping: &[usize],
    group_sizes: &[usize],
    num_groups: usize,
) -> f64 {
    let mut s_t_doubled = 0.0;
    let mut s_w = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d2 = condensed_dist(condensed, n, i, j).powi(2);
            s_t_doubled += d2;
            if grouping[i] == grouping[j] {
                s_w += d2 / group_sizes[grouping[i]] as f64;
            }
        }
    }
    let s_t = s_t_doubled / n as f64;
    let s_a = s_t - s_w;
    (s_a / (num_groups - 1) as f64) / (s_w / (n - num_groups) as f64)
}

/// Relabel arbitrary group ids into dense `0..num_groups` indices, ordered by
/// the sorted distinct id values (matches `np.unique(grouping,
/// return_inverse=True)`).
fn dense_group_indices(cluster_ids: &[u32]) -> (Vec<usize>, usize) {
    let mut sorted: Vec<u32> = cluster_ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let index_of: HashMap<u32, usize> = sorted.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let dense = cluster_ids.iter().map(|c| index_of[c]).collect();
    (dense, sorted.len())
}

/// Build a PermanovaResultsSchema DataFrame from a precomputed condensed
/// distance matrix (over geocodes, in `geocode_ids` order) and a geocode ->
/// cluster mapping.
///
/// Mirrors `build_permanova_results_df`. `geocode_distance_matrix` is passed
/// in as its condensed form directly rather than as a `GeocodeDistanceMatrix`
/// object, since building that matrix involves UMAP (Phase 3, stays in
/// Python) — this function only needs the already-computed distances.
#[pyfunction]
#[pyo3(signature = (condensed, geocode_ids, geocode_cluster_df, permutations = 999))]
pub fn build_permanova_results(
    condensed: Vec<f64>,
    geocode_ids: Vec<u64>,
    geocode_cluster_df: PyDataFrame,
    permutations: u64,
) -> PyResult<PyDataFrame> {
    let geocode_cluster_df: DataFrame = geocode_cluster_df.into();
    let n = geocode_ids.len();
    if condensed.len() != n * (n.saturating_sub(1)) / 2 {
        return Err(PyValueError::new_err(format!(
            "condensed distance vector length {} doesn't match {} geocodes",
            condensed.len(),
            n
        )));
    }

    let geocode_to_cluster: HashMap<u64, u32> = {
        let geocode = geocode_cluster_df
            .column("geocode")
            .map_err(to_py)?
            .as_materialized_series()
            .u64()
            .map_err(to_py)?
            .clone();
        let cluster = geocode_cluster_df
            .column("cluster")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        geocode
            .into_no_null_iter()
            .zip(cluster.into_no_null_iter())
            .collect()
    };

    let cluster_ids: Vec<u32> = geocode_ids
        .iter()
        .map(|g| {
            geocode_to_cluster.get(g).copied().ok_or_else(|| {
                PyValueError::new_err(format!("geocode {g} missing from geocode_cluster_df"))
            })
        })
        .collect::<PyResult<_>>()?;

    let (grouping, num_groups) = dense_group_indices(&cluster_ids);
    if num_groups == grouping.len() {
        return Err(PyValueError::new_err(
            "all values in the grouping vector are unique",
        ));
    }
    if num_groups < 2 {
        return Err(PyValueError::new_err(
            "all values in the grouping vector are the same",
        ));
    }

    let mut group_sizes = vec![0usize; num_groups];
    for &g in &grouping {
        group_sizes[g] += 1;
    }

    let test_statistic = permanova_f_stat(&condensed, n, &grouping, &group_sizes, num_groups);

    let p_value = if permutations > 0 {
        let mut rng = rng();
        let mut perm = grouping.clone();
        let mut count = 0u64;
        for _ in 0..permutations {
            perm.shuffle(&mut rng);
            if permanova_f_stat(&condensed, n, &perm, &group_sizes, num_groups) >= test_statistic {
                count += 1;
            }
        }
        (count as f64 + 1.0) / (permutations as f64 + 1.0)
    } else {
        f64::NAN
    };

    let out = DataFrame::new(
        1,
        vec![
            StringChunked::from_iter_values("method_name".into(), std::iter::once("PERMANOVA"))
                .into_column(),
            StringChunked::from_iter_values(
                "test_statistic_name".into(),
                std::iter::once("pseudo-F"),
            )
            .into_column(),
            Float64Chunked::from_vec("test_statistic".into(), vec![test_statistic]).into_column(),
            Float64Chunked::from_vec("p_value".into(), vec![p_value]).into_column(),
            UInt64Chunked::from_vec("permutations".into(), vec![permutations]).into_column(),
        ],
    )
    .map_err(to_py)?;

    Ok(PyDataFrame(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condensed_dist_matches_squareform_layout() {
        // 4x4 squareform:
        //      0   1   2   3
        //  0 [ 0,  1,  2,  3]
        //  1 [ 1,  0,  4,  5]
        //  2 [ 2,  4,  0,  6]
        //  3 [ 3,  5,  6,  0]
        // condensed (scipy order): (0,1)=1 (0,2)=2 (0,3)=3 (1,2)=4 (1,3)=5 (2,3)=6
        let condensed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(condensed_dist(&condensed, 4, 0, 1), 1.0);
        assert_eq!(condensed_dist(&condensed, 4, 1, 0), 1.0);
        assert_eq!(condensed_dist(&condensed, 4, 2, 3), 6.0);
        assert_eq!(condensed_dist(&condensed, 4, 1, 2), 4.0);
    }

    #[test]
    fn f_stat_matches_skbio_on_all_equal_distances() {
        // permanova(DistanceMatrix([1]*6, ids=[a,b,c,d]), [0,0,1,1],
        // permutations=0).statistic == 1.0
        let condensed = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let grouping = [0, 0, 1, 1];
        let group_sizes = [2, 2];
        let f = permanova_f_stat(&condensed, 4, &grouping, &group_sizes, 2);
        assert!((f - 1.0).abs() < 1e-12);
    }
}
