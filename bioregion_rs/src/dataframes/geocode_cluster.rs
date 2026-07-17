//! Port of `src/dataframes/geocode_cluster.py`'s per-k clustering loop
//! (`build_geocode_cluster_multi_k_df`).
//!
//! The Python code calls, once per `k` in `[min_k, max_k]`:
//!
//! ```python
//! AgglomerativeClustering(
//!     n_clusters=k,
//!     connectivity=csr_matrix(connectivity_matrix),
//!     linkage="ward",
//! ).fit_predict(distance_matrix.squareform())
//! ```
//!
//! There is no Rust library that does connectivity-constrained Ward
//! agglomerative clustering, so this reproduces sklearn's own algorithm
//! exactly, reading straight from its source rather than its docs:
//! `sklearn.cluster._agglomerative.ward_tree` (the connectivity-constrained
//! hierarchical merge) and `_hc_cut` (cutting the resulting full tree at `k`
//! clusters), plus the two Cython helpers `compute_ward_dist` and
//! `_get_parents` from `_hierarchical_fast.pyx`.
//!
//! Two facts about that algorithm make an exact, deterministic port possible:
//!
//! 1. Note the Python passes the *squareform distance matrix* as `X` with the
//!    default `metric="euclidean"` (not `metric="precomputed"`). So Ward treats
//!    each row of the square distance matrix as an n-dimensional feature vector
//!    and computes ordinary Euclidean distances between rows — the same
//!    "unusual but pre-existing" methodology already preserved in
//!    `geocode_cluster_metrics.rs`. `feature vector for point i` is therefore
//!    exactly `condensed_dist(.., i, j) for j in 0..n`.
//! 2. sklearn's merge heap is keyed on the tuple `(inertia, row, col)` where
//!    every `(row, col)` pair is pushed at most once and every pair is distinct,
//!    so the key is a *strict total order* with no ties. The merge sequence is
//!    thus uniquely determined and independent of any heap's internal layout —
//!    a plain `BinaryHeap` reproduces sklearn's `heapq` pop order bit-for-bit.
//!
//! The one place a heap's internal array order is observable is `_hc_cut`'s
//! final `enumerate(nodes)` (it decides which cluster gets integer label 0, 1,
//! …), so that step reimplements CPython's `heapq` push/pushpop sift operations
//! precisely, to reproduce sklearn's cluster-label integers and not merely an
//! equivalent partition.
//!
//! Building the full tree once and cutting it at every `k` (as `compute_full_tree`
//! does — always true here since `k` is far below `max(100, 0.02 * n_samples)`)
//! also means we run the O(n^2 * features) merge a single time instead of once
//! per `k` as the Python loop does.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::dataframes::geocode_cluster_metrics::condensed_dist;
use crate::to_py;

/// A candidate merge between clusters `hi` and `lo` (`hi > lo`, matching how
/// sklearn stores `(coord_row, coord_col)`), ordered ascending by
/// `(weight, hi, lo)` so a min-heap reproduces `heapq`'s pop order exactly.
#[derive(Clone, Copy)]
struct Edge {
    weight: f64,
    hi: usize,
    lo: usize,
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Edge {}
impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight
            .total_cmp(&other.weight)
            .then(self.hi.cmp(&other.hi))
            .then(self.lo.cmp(&other.lo))
    }
}
impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Ward inertia of merging clusters `row` and `col`, mirroring the arithmetic
/// (and summation order) of sklearn's Cython `compute_ward_dist`.
fn ward_dist(m1: &[f64], m2: &[Vec<f64>], row: usize, col: usize) -> f64 {
    let (mr, mc) = (m1[row], m1[col]);
    let n = (mr * mc) / (mr + mc);
    let mut pa = 0.0;
    let (r2, c2) = (&m2[row], &m2[col]);
    for f in 0..r2.len() {
        let d = r2[f] / mr - c2[f] / mc;
        pa += d * d;
    }
    pa * n
}

/// Build the full connectivity-constrained Ward tree, returning `children`
/// in sklearn's format: `children[m] == (lower, higher)` are the two nodes
/// merged to form node `n + m`. `edges` are the undirected connectivity edges
/// as `(a, b)` index pairs (each edge once; direction irrelevant).
fn ward_tree(condensed: &[f64], n: usize, edges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let n_nodes = 2 * n - 1;

    // moments_1 = cluster sizes (leaves = 1); moments_2 = summed feature rows
    // (leaf i's features are row i of the squareform distance matrix).
    let mut m1 = vec![0.0f64; n_nodes];
    for v in m1.iter_mut().take(n) {
        *v = 1.0;
    }
    let mut m2: Vec<Vec<f64>> = vec![Vec::new(); n_nodes];
    for (i, slot) in m2.iter_mut().enumerate().take(n) {
        *slot = (0..n).map(|j| condensed_dist(condensed, n, i, j)).collect();
    }

    // Adjacency ("structure matrix" A). Symmetric: every edge on both endpoints.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for &(a, b) in edges {
        adj[a].push(b);
        adj[b].push(a);
    }

    // Seed the heap with one entry per undirected edge, keyed (inertia, hi, lo).
    let mut heap: BinaryHeap<Reverse<Edge>> = BinaryHeap::new();
    for &(a, b) in edges {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let weight = ward_dist(&m1, &m2, hi, lo);
        heap.push(Reverse(Edge { weight, hi, lo }));
    }

    let mut parent: Vec<usize> = (0..n_nodes).collect();
    let mut used = vec![true; n_nodes];
    let mut not_visited = vec![true; n_nodes];
    let mut children: Vec<(usize, usize)> = Vec::with_capacity(n - 1);

    for k in n..n_nodes {
        // Pop the cheapest still-valid merge (both endpoints not yet merged).
        let (i, j) = loop {
            let Reverse(e) = heap.pop().expect("heap exhausted before tree complete");
            if used[e.hi] && used[e.lo] {
                break (e.hi, e.lo);
            }
        };
        parent[i] = k;
        parent[j] = k;
        children.push((i, j)); // (hi, lo); reversed to (lo, hi) on return
        used[i] = false;
        used[j] = false;

        // Update moments for the new node, then free the two merged rows — they
        // are never read again (only current roots are ever revisited), which
        // halves peak feature-row memory versus sklearn's static moments array.
        m1[k] = m1[i] + m1[j];
        let row: Vec<f64> = (0..m2[i].len()).map(|f| m2[i][f] + m2[j][f]).collect();
        m2[i] = Vec::new();
        m2[j] = Vec::new();
        m2[k] = row;

        // Recompute the structure matrix: the distinct current roots reachable
        // from the neighbours of i then j (mirrors `_get_parents` over A[i], A[j]).
        for v in not_visited.iter_mut() {
            *v = true;
        }
        not_visited[k] = false;
        let mut coord_col: Vec<usize> = Vec::new();
        for &start in adj[i].iter().chain(adj[j].iter()) {
            let mut node = start;
            while parent[node] != node {
                node = parent[node];
            }
            if not_visited[node] {
                not_visited[node] = false;
                coord_col.push(node);
            }
        }
        for &col in &coord_col {
            adj[col].push(k);
            let weight = ward_dist(&m1, &m2, k, col);
            heap.push(Reverse(Edge {
                weight,
                hi: k,
                lo: col,
            }));
        }
        adj[k] = coord_col;
    }

    children.into_iter().map(|(hi, lo)| (lo, hi)).collect()
}

// --- CPython `heapq`, reproduced exactly for `_hc_cut` label numbering -------
//
// Operates on negated node ids (so the min-heap yields the largest id), matching
// sklearn's `nodes = [-(...)]` convention. Only the operations `_hc_cut` uses
// (push, pushpop) are implemented.

fn heap_siftdown(heap: &mut [i64], startpos: usize, mut pos: usize) {
    let newitem = heap[pos];
    while pos > startpos {
        let parentpos = (pos - 1) >> 1;
        if newitem < heap[parentpos] {
            heap[pos] = heap[parentpos];
            pos = parentpos;
        } else {
            break;
        }
    }
    heap[pos] = newitem;
}

fn heap_siftup(heap: &mut [i64], mut pos: usize) {
    let endpos = heap.len();
    let startpos = pos;
    let newitem = heap[pos];
    let mut childpos = 2 * pos + 1;
    while childpos < endpos {
        let rightpos = childpos + 1;
        if rightpos < endpos && !(heap[childpos] < heap[rightpos]) {
            childpos = rightpos;
        }
        heap[pos] = heap[childpos];
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    heap[pos] = newitem;
    heap_siftdown(heap, startpos, pos);
}

fn heap_push(heap: &mut Vec<i64>, item: i64) {
    heap.push(item);
    let last = heap.len() - 1;
    heap_siftdown(heap, 0, last);
}

fn heap_pushpop(heap: &mut [i64], mut item: i64) -> i64 {
    if !heap.is_empty() && heap[0] < item {
        std::mem::swap(&mut heap[0], &mut item);
        heap_siftup(heap, 0);
    }
    item
}

/// All leaf (`< n_leaves`) descendants of `node` — mirrors
/// `_hierarchical._hc_get_descendent` (the leaf set is order-independent).
fn descendents(node: usize, children: &[(usize, usize)], n_leaves: usize) -> Vec<usize> {
    if node < n_leaves {
        return vec![node];
    }
    let mut stack = vec![node];
    let mut out = Vec::new();
    while let Some(i) = stack.pop() {
        if i < n_leaves {
            out.push(i);
        } else {
            let (a, b) = children[i - n_leaves];
            stack.push(a);
            stack.push(b);
        }
    }
    out
}

/// Cut the full Ward tree into `k` clusters, returning per-leaf labels.
/// Reproduces `sklearn.cluster._agglomerative._hc_cut`, including the exact
/// `heapq` array order that fixes each cluster's integer label.
fn hc_cut(k: usize, children: &[(usize, usize)], n_leaves: usize) -> Vec<u32> {
    // Root id == n_leaves + (#merges) - 1; the second-to-last created node is
    // always a child of the root, so `max(children[-1]) + 1 == root` (see the
    // sklearn source), which is what this mirrors.
    let root = n_leaves + children.len() - 1;
    let mut nodes: Vec<i64> = vec![-(root as i64)];
    for _ in 0..k.saturating_sub(1) {
        let node_id = (-nodes[0]) as usize;
        let (c0, c1) = children[node_id - n_leaves];
        heap_push(&mut nodes, -(c0 as i64));
        heap_pushpop(&mut nodes, -(c1 as i64));
    }

    let mut label = vec![0u32; n_leaves];
    for (i, &node) in nodes.iter().enumerate() {
        let head = (-node) as usize;
        for leaf in descendents(head, children, n_leaves) {
            label[leaf] = i as u32;
        }
    }
    label
}

/// Connectivity-constrained Ward clustering for every `k` in `[min_k, max_k]`.
///
/// `geocodes` are the row labels of the distance/connectivity matrices (same
/// order for all three). `condensed` is the scipy `pdist`-order upper triangle
/// of the squareform distance matrix used as Ward's feature matrix `X`.
/// `edge_a`/`edge_b` are the connectivity graph's undirected edges as index
/// pairs into that ordering. Returns one row per (geocode, k) with the cluster
/// label, matching `GeocodeClusterMultiKSchema`.
#[pyfunction]
#[pyo3(signature = (geocodes, condensed, edge_a, edge_b, min_k, max_k))]
pub fn build_geocode_cluster_multi_k(
    geocodes: Vec<u64>,
    condensed: Vec<f64>,
    edge_a: Vec<usize>,
    edge_b: Vec<usize>,
    min_k: usize,
    max_k: usize,
) -> PyResult<PyDataFrame> {
    let n = geocodes.len();
    let edges: Vec<(usize, usize)> = edge_a
        .iter()
        .zip(edge_b.iter())
        .map(|(&a, &b)| (a, b))
        .collect();

    let children = ward_tree(&condensed, n, &edges);

    let num_ks = if max_k >= min_k { max_k - min_k + 1 } else { 0 };
    let total = n * num_ks;
    let mut geocode_col: Vec<u64> = Vec::with_capacity(total);
    let mut num_clusters_col: Vec<u32> = Vec::with_capacity(total);
    let mut cluster_col: Vec<u32> = Vec::with_capacity(total);
    for k in min_k..=max_k {
        let labels = hc_cut(k, &children, n);
        for idx in 0..n {
            geocode_col.push(geocodes[idx]);
            num_clusters_col.push(k as u32);
            cluster_col.push(labels[idx]);
        }
    }

    let out = DataFrame::new(
        total,
        vec![
            UInt64Chunked::from_vec("geocode".into(), geocode_col).into_column(),
            UInt32Chunked::from_vec("num_clusters".into(), num_clusters_col).into_column(),
            UInt32Chunked::from_vec("cluster".into(), cluster_col).into_column(),
        ],
    )
    .map_err(to_py)?;
    Ok(PyDataFrame(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny path graph 0-1-2-3-4 with distances that make the ends closer to
    // their inner neighbours; exercises the merge loop and the tree cut.
    fn path_edges(n: usize) -> Vec<(usize, usize)> {
        (0..n - 1).map(|i| (i, i + 1)).collect()
    }

    fn condensed_from_square(sq: &[Vec<f64>]) -> Vec<f64> {
        let n = sq.len();
        let mut c = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                c.push(sq[i][j]);
            }
        }
        c
    }

    #[test]
    fn full_tree_has_n_minus_1_merges() {
        let n = 5;
        // simple metric: |i - j|
        let sq: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| (i as f64 - j as f64).abs()).collect())
            .collect();
        let condensed = condensed_from_square(&sq);
        let children = ward_tree(&condensed, n, &path_edges(n));
        assert_eq!(children.len(), n - 1);
        // every child is stored (lower, higher)
        for &(lo, hi) in &children {
            assert!(lo < hi);
        }
    }

    #[test]
    fn cut_produces_requested_cluster_count_and_covers_all_leaves() {
        let n = 6;
        let sq: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| (i as f64 - j as f64).abs()).collect())
            .collect();
        let condensed = condensed_from_square(&sq);
        let children = ward_tree(&condensed, n, &path_edges(n));
        for k in 2..=4 {
            let labels = hc_cut(k, &children, n);
            assert_eq!(labels.len(), n);
            let distinct: std::collections::HashSet<u32> = labels.iter().copied().collect();
            assert_eq!(distinct.len(), k, "k={k} should yield k clusters");
            // labels are a dense 0..k range
            assert_eq!(
                distinct,
                (0..k as u32).collect::<std::collections::HashSet<_>>()
            );
        }
        // On a path graph, contiguous points should cluster together.
        let labels2 = hc_cut(2, &children, n);
        // the split of a path into 2 connectivity-constrained ward clusters is
        // a single cut point: labels form two contiguous runs.
        let mut changes = 0;
        for w in labels2.windows(2) {
            if w[0] != w[1] {
                changes += 1;
            }
        }
        assert_eq!(changes, 1, "a path split into 2 clusters has one boundary");
    }

    #[test]
    fn cpython_heapq_pushpop_matches_reference() {
        // Reproduce a small sequence and compare against hand-computed heapq order.
        let mut h: Vec<i64> = Vec::new();
        for v in [-4, -1, -3, -2, -5] {
            heap_push(&mut h, v);
        }
        // CPython heapq internal array for these pushes:
        // push -4 -> [-4]
        // push -1 -> [-4,-1]
        // push -3 -> [-4,-1,-3]
        // push -2 -> [-4,-2,-3,-1]
        // push -5 -> [-5,-4,-3,-1,-2]
        assert_eq!(h, vec![-5, -4, -3, -1, -2]);
        let popped = heap_pushpop(&mut h, -6);
        assert_eq!(popped, -6, "pushpop of a smaller item returns it unchanged");
        let popped = heap_pushpop(&mut h, 0);
        assert_eq!(popped, -5, "pushpop of a larger item evicts the min");
    }
}
