//! Port of `src/dataframes/cluster_significant_differences.py`
//! (`build_cluster_significant_differences_df`).

use std::collections::{BTreeSet, HashMap};

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::to_py;

const P_VALUE_THRESHOLD: f64 = 0.05;
const MIN_COUNT_THRESHOLD: u32 = 5;

// --- Fisher's exact test (2x2, two-sided) -----------------------------------
//
// A from-scratch port of `scipy.stats.fisher_exact(table, alternative="two-sided")`
// for 2x2 tables: the null hypothesis is that the input table is drawn from a
// hypergeometric distribution with the observed margins fixed, and the
// two-sided p-value sums the probability of every table at least as extreme
// (i.e. with pmf <= the observed table's pmf, up to a small tolerance) as the
// one observed. Mirrors scipy's actual 2x2 algorithm (mode + binary search on
// the tail), not the "sum every point in the support" description in its
// docstring, which is only how the result is *explained*, not computed.

/// Lanczos approximation of ln(Γ(x)). Only ever called with x >= 1 in this
/// module (n+1 for n >= 0), so the reflection formula for x < 0.5 isn't needed.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEFFICIENTS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let t = x + G + 0.5;
    let mut a = COEFFICIENTS[0];
    for (i, &c) in COEFFICIENTS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// ln(C(n, k)), or -infinity if k is outside [0, n].
fn ln_choose(n: i64, k: i64) -> f64 {
    if k < 0 || k > n {
        return f64::NEG_INFINITY;
    }
    ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64)
}

/// Support of `hypergeom_pmf(_, m, n_pop, n_draw)`: `[lo, hi]` inclusive.
fn hypergeom_support(m: i64, n_pop: i64, n_draw: i64) -> (i64, i64) {
    (0.max(n_draw - (m - n_pop)), n_pop.min(n_draw))
}

/// `scipy.stats.hypergeom.pmf(x, m, n_pop, n_draw)`.
fn hypergeom_pmf(x: i64, m: i64, n_pop: i64, n_draw: i64) -> f64 {
    let (lo, hi) = hypergeom_support(m, n_pop, n_draw);
    if x < lo || x > hi {
        return 0.0;
    }
    (ln_choose(n_pop, x) + ln_choose(m - n_pop, n_draw - x) - ln_choose(m, n_draw)).exp()
}

/// `scipy.stats.hypergeom.cdf(x, m, n_pop, n_draw)` = P(X <= x).
fn hypergeom_cdf(x: i64, m: i64, n_pop: i64, n_draw: i64) -> f64 {
    let (lo, _) = hypergeom_support(m, n_pop, n_draw);
    (lo..=x).map(|k| hypergeom_pmf(k, m, n_pop, n_draw)).sum()
}

/// `scipy.stats.hypergeom.sf(x, m, n_pop, n_draw)` = P(X > x).
fn hypergeom_sf(x: i64, m: i64, n_pop: i64, n_draw: i64) -> f64 {
    let (_, hi) = hypergeom_support(m, n_pop, n_draw);
    ((x + 1)..=hi)
        .map(|k| hypergeom_pmf(k, m, n_pop, n_draw))
        .sum()
}

/// Mirrors scipy's `_binary_search_for_binom_tst`: for `a` ascending over
/// `[lo, hi]`, find the largest `i` such that `a(i) <= d`.
fn binary_search(a: impl Fn(i64) -> f64, d: f64, lo: i64, hi: i64) -> i64 {
    let (mut lo, mut hi) = (lo, hi);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let midval = a(mid);
        if midval < d {
            lo = mid + 1;
        } else if midval > d {
            hi = mid - 1;
        } else {
            return mid;
        }
    }
    if a(lo) <= d { lo } else { lo - 1 }
}

/// Two-sided Fisher's exact test p-value for the 2x2 table `[[a, b], [c, d]]`.
/// Mirrors the `alternative == "two-sided"` branch of `scipy.stats.fisher_exact`.
fn fisher_exact_two_sided_pvalue(a: i64, b: i64, c: i64, d: i64) -> f64 {
    // Matches scipy's early return: a zero row or column sum makes the table
    // degenerate (every table with these margins is the same table).
    if a + b == 0 || c + d == 0 || a + c == 0 || b + d == 0 {
        return 1.0;
    }

    let n1 = a + b;
    let n2 = c + d;
    let n = a + c;
    let m = n1 + n2;
    let pmf = |x: i64| hypergeom_pmf(x, m, n1, n);

    let mode = ((n + 1) * (n1 + 1)) / (n1 + n2 + 2);
    let pexact = pmf(a);
    let pmode = pmf(mode);

    let epsilon = 1e-14;
    let gamma = 1.0 + epsilon;

    if (pexact - pmode).abs() / pexact.max(pmode) <= epsilon {
        return 1.0;
    }

    let pvalue = if a < mode {
        let plower = hypergeom_cdf(a, m, n1, n);
        if hypergeom_pmf(n, m, n1, n) > pexact * gamma {
            plower
        } else {
            let guess = binary_search(|x| -pmf(x), -pexact * gamma, mode, n);
            plower + hypergeom_sf(guess, m, n1, n)
        }
    } else {
        let pupper = hypergeom_sf(a - 1, m, n1, n);
        if hypergeom_pmf(0, m, n1, n) > pexact * gamma {
            pupper
        } else {
            let guess = binary_search(pmf, pexact * gamma, 0, mode);
            pupper + hypergeom_cdf(guess, m, n1, n)
        }
    };

    pvalue.min(1.0)
}

// --- build_cluster_significant_differences_df -------------------------------

struct Row {
    cluster: u32,
    taxon_id: u32,
    p_value: f64,
    log2_fold_change: f64,
    cluster_count: u32,
    neighbor_count: u32,
}

/// Build a ClusterSignificantDifferencesSchema DataFrame: taxa whose
/// occurrence rate significantly differs (Fisher's exact, two-sided) between
/// a cluster and its neighbors, with normalized scores highlighting taxa that
/// are both distinctive and common.
///
/// Mirrors `build_cluster_significant_differences_df`.
#[pyfunction]
pub fn build_cluster_significant_differences(
    all_stats: PyDataFrame,
    cluster_neighbors: PyDataFrame,
) -> PyResult<PyDataFrame> {
    let all_stats: DataFrame = all_stats.into();
    let cluster_neighbors: DataFrame = cluster_neighbors.into();

    let neighbors_map: HashMap<u32, Vec<u32>> = {
        let cluster_ca = cluster_neighbors
            .column("cluster")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let neighbors_ca = cluster_neighbors
            .column("direct_and_indirect_neighbors")
            .map_err(to_py)?
            .as_materialized_series()
            .list()
            .map_err(to_py)?
            .clone();
        let mut map = HashMap::new();
        for (i, cluster) in cluster_ca.into_no_null_iter().enumerate() {
            let neighbors: Vec<u32> = neighbors_ca
                .get_as_series(i)
                .map(|s| {
                    s.u32()
                        .expect("neighbor list must be UInt32")
                        .into_no_null_iter()
                        .collect()
                })
                .unwrap_or_default();
            map.insert(cluster, neighbors);
        }
        map
    };

    let stats: Vec<(Option<u32>, u32, u32)> = {
        let cluster_ca = all_stats
            .column("cluster")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let taxon_ca = all_stats
            .column("taxonId")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let count_ca = all_stats
            .column("count")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        cluster_ca
            .iter()
            .zip(taxon_ca.into_no_null_iter())
            .zip(count_ca.into_no_null_iter())
            .map(|((cluster, taxon_id), count)| (cluster, taxon_id, count))
            .collect()
    };

    let cluster_ids: BTreeSet<u32> = stats.iter().filter_map(|&(c, _, _)| c).collect();

    let mut rows: Vec<Row> = Vec::new();
    for cluster in cluster_ids {
        let Some(neighbors) = neighbors_map.get(&cluster) else {
            continue;
        };
        if neighbors.is_empty() {
            continue;
        }
        let neighbor_set: std::collections::HashSet<u32> = neighbors.iter().copied().collect();

        let cluster_stats: Vec<(u32, u32)> = stats
            .iter()
            .filter(|&&(c, _, _)| c == Some(cluster))
            .map(|&(_, t, cnt)| (t, cnt))
            .collect();
        let mut neighbor_counts: HashMap<u32, u32> = HashMap::new();
        for &(c, t, cnt) in &stats {
            if c.is_some_and(|c| neighbor_set.contains(&c)) {
                *neighbor_counts.entry(t).or_default() += cnt;
            }
        }

        let total_cluster_count: u32 = cluster_stats.iter().map(|&(_, c)| c).sum();
        let total_neighbor_count: u32 = neighbor_counts.values().sum();

        for (taxon_id, count) in cluster_stats {
            if count < MIN_COUNT_THRESHOLD {
                continue;
            }
            let neighbor_count = neighbor_counts.get(&taxon_id).copied().unwrap_or(0);
            if neighbor_count < MIN_COUNT_THRESHOLD {
                continue;
            }

            let p_value = fisher_exact_two_sided_pvalue(
                i64::from(count),
                i64::from(total_cluster_count - count),
                i64::from(neighbor_count),
                i64::from(total_neighbor_count - neighbor_count),
            );
            if p_value >= P_VALUE_THRESHOLD {
                continue;
            }

            let mean_cluster = f64::from(count) / f64::from(total_cluster_count);
            let mean_neighbor = f64::from(neighbor_count) / f64::from(total_neighbor_count);
            if mean_cluster == 0.0 || mean_neighbor == 0.0 {
                continue;
            }
            let log2_fold_change = (mean_cluster / mean_neighbor).log2();

            rows.push(Row {
                cluster,
                taxon_id,
                p_value,
                log2_fold_change,
                cluster_count: count,
                neighbor_count,
            });
        }
    }

    let (high_scores, low_scores) = score_rows(&rows);

    let n = rows.len();
    let mut clusters = Vec::with_capacity(n);
    let mut taxon_ids = Vec::with_capacity(n);
    let mut p_values = Vec::with_capacity(n);
    let mut log2_fold_changes = Vec::with_capacity(n);
    let mut cluster_counts = Vec::with_capacity(n);
    let mut neighbor_counts_out = Vec::with_capacity(n);
    for row in &rows {
        clusters.push(row.cluster);
        taxon_ids.push(row.taxon_id);
        p_values.push(row.p_value);
        log2_fold_changes.push(row.log2_fold_change);
        cluster_counts.push(row.cluster_count);
        neighbor_counts_out.push(row.neighbor_count);
    }

    let out = DataFrame::new(
        n,
        vec![
            UInt32Chunked::from_vec("cluster".into(), clusters).into_column(),
            UInt32Chunked::from_vec("taxonId".into(), taxon_ids).into_column(),
            Float64Chunked::from_vec("p_value".into(), p_values).into_column(),
            Float64Chunked::from_vec("log2_fold_change".into(), log2_fold_changes).into_column(),
            UInt32Chunked::from_vec("cluster_count".into(), cluster_counts).into_column(),
            UInt32Chunked::from_vec("neighbor_count".into(), neighbor_counts_out).into_column(),
            Float64Chunked::from_vec("high_log2_high_count_score".into(), high_scores)
                .into_column(),
            Float64Chunked::from_vec("low_log2_high_count_score".into(), low_scores).into_column(),
        ],
    )
    .map_err(to_py)?;

    Ok(PyDataFrame(out))
}

/// Normalize `log2_fold_change` (linear) and `cluster_count` (logarithmic)
/// to [0, 1] across all rows, then compute the two composite scores. Returns
/// all-zero scores if there aren't at least two distinct values to normalize
/// against (matches Python's fallback for an empty/degenerate range).
fn score_rows(rows: &[Row]) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let log2_min = rows
        .iter()
        .map(|r| r.log2_fold_change)
        .fold(f64::INFINITY, f64::min);
    let log2_max = rows
        .iter()
        .map(|r| r.log2_fold_change)
        .fold(f64::NEG_INFINITY, f64::max);
    let cluster_min = rows.iter().map(|r| r.cluster_count).min().unwrap();
    let cluster_max = rows.iter().map(|r| r.cluster_count).max().unwrap();

    if log2_min == log2_max || cluster_min == cluster_max {
        return (vec![0.0; n], vec![0.0; n]);
    }

    let ln_cluster_min = f64::from(cluster_min).ln();
    let ln_cluster_max = f64::from(cluster_max).ln();

    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    for row in rows {
        let log2_norm = (row.log2_fold_change - log2_min) / (log2_max - log2_min);
        let cluster_norm = (f64::from(row.cluster_count).ln() - ln_cluster_min)
            / (ln_cluster_max - ln_cluster_min);
        high.push(log2_norm * cluster_norm);
        low.push(if row.log2_fold_change < 0.0 {
            (1.0 - log2_norm) * cluster_norm
        } else {
            0.0
        });
    }
    (high, low)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisher_exact_matches_scipy_docstring_examples() {
        // scipy.stats.fisher_exact([[6, 2], [1, 4]]).pvalue == 0.10256410256410257
        assert!((fisher_exact_two_sided_pvalue(6, 2, 1, 4) - 0.10256410256410257).abs() < 1e-12);
        // scipy.stats.fisher_exact([[8, 2], [1, 5]]).pvalue == 0.034965034965034975
        assert!((fisher_exact_two_sided_pvalue(8, 2, 1, 5) - 0.034965034965034975).abs() < 1e-12);
    }

    #[test]
    fn fisher_exact_zero_margin_is_not_significant() {
        assert_eq!(fisher_exact_two_sided_pvalue(5, 0, 5, 0), 1.0);
    }
}
