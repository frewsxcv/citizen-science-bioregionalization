//! Port of `src/dataframes/cluster_color.py` — **geographic path only**
//! (`_build_geographic`). The taxonomic path uses UMAP + MDS and stays in
//! Python (Phase 3).

use std::collections::{HashMap, HashSet};

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::colors::darken_hex_color;
use crate::to_py;

/// Control points (x, y) for matplotlib's `YlOrRd` colormap channels, copied
/// from `matplotlib.colormaps['YlOrRd']._segmentdata` (a `LinearSegmentedColormap`
/// with no discontinuities, so y0 == y1 at every stop).
const YLORRD_RED: [(f64, f64); 9] = [
    (0.0, 1.0),
    (0.125, 1.0),
    (0.25, 0.996078431372549),
    (0.375, 0.996078431372549),
    (0.5, 0.9921568627450981),
    (0.625, 0.9882352941176471),
    (0.75, 0.8901960784313725),
    (0.875, 0.7411764705882353),
    (1.0, 0.5019607843137255),
];
const YLORRD_GREEN: [(f64, f64); 9] = [
    (0.0, 1.0),
    (0.125, 0.9294117647058824),
    (0.25, 0.8509803921568627),
    (0.375, 0.6980392156862745),
    (0.5, 0.5529411764705883),
    (0.625, 0.3058823529411765),
    (0.75, 0.10196078431372549),
    (0.875, 0.0),
    (1.0, 0.0),
];
const YLORRD_BLUE: [(f64, f64); 9] = [
    (0.0, 0.8),
    (0.125, 0.6274509803921569),
    (0.25, 0.4627450980392157),
    (0.375, 0.2980392156862745),
    (0.5, 0.23529411764705882),
    (0.625, 0.16470588235294117),
    (0.75, 0.10980392156862745),
    (0.875, 0.14901960784313725),
    (1.0, 0.14901960784313725),
];

/// Piecewise-linear interpolation over a sorted set of (x, y) control points.
fn interp(points: &[(f64, f64)], t: f64) -> f64 {
    for w in points.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if t <= x1 {
            let frac = if x1 > x0 { (t - x0) / (x1 - x0) } else { 0.0 };
            return y0 + frac * (y1 - y0);
        }
    }
    points.last().unwrap().1
}

/// Replicates `sns.color_palette("YlOrRd", n).as_hex()`: matplotlib samples a
/// `LinearSegmentedColormap` by building a 256-entry lookup table (linear
/// interpolation of the control points at 256 evenly spaced positions) and
/// then indexing it with `floor(x * 256)`; seaborn picks the `n` sample
/// positions as `linspace(0, 1, n + 2)[1:-1]` (excluding the extremes "to
/// provide better contrast"). Verified bit-for-bit against
/// `sns.color_palette("YlOrRd", n).as_hex()` for n up to 15 while porting.
fn ylorrd_hex_palette(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let x = (i as f64 + 1.0) / (n as f64 + 1.0);
            let idx = ((x * 256.0).floor() as usize).min(255);
            let t = idx as f64 / 255.0;
            let to_u8 = |v: f64| (v * 255.0).round() as u8;
            format!(
                "#{:02x}{:02x}{:02x}",
                to_u8(interp(&YLORRD_RED, t)),
                to_u8(interp(&YLORRD_GREEN, t)),
                to_u8(interp(&YLORRD_BLUE, t)),
            )
        })
        .collect()
}

/// Greedy graph coloring with the `largest_first` node ordering (nodes sorted
/// by degree descending, ties broken by original order — a stable sort, same
/// as Python's `sorted(G, key=G.degree, reverse=True)`), assigning each node
/// the smallest color index not already used by a colored neighbor. Mirrors
/// `networkx.coloring.greedy_color(G, strategy="largest_first")`.
fn greedy_color(cluster_ids: &[u32], adjacency: &HashMap<u32, HashSet<u32>>) -> HashMap<u32, u32> {
    let mut order = cluster_ids.to_vec();
    order.sort_by_key(|c| std::cmp::Reverse(adjacency.get(c).map_or(0, HashSet::len)));

    let mut colors: HashMap<u32, u32> = HashMap::new();
    for node in order {
        let neighbor_colors: HashSet<u32> = adjacency
            .get(&node)
            .into_iter()
            .flatten()
            .filter_map(|n| colors.get(n).copied())
            .collect();
        let mut color = 0;
        while neighbor_colors.contains(&color) {
            color += 1;
        }
        colors.insert(node, color);
    }
    colors
}

/// Build a ClusterColorSchema DataFrame using geographic neighbor-based
/// coloring: greedy-color the cluster adjacency graph so neighboring
/// clusters get different colors, then map color classes to a `YlOrRd` hex
/// palette (fewest colors needed, in class order).
///
/// Mirrors `build_cluster_color_df(..., color_method="geographic")` /
/// `_build_geographic`.
#[pyfunction]
pub fn build_cluster_color(cluster_neighbors_df: PyDataFrame) -> PyResult<PyDataFrame> {
    let df: DataFrame = cluster_neighbors_df.into();

    let cluster_ca = df
        .column("cluster")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .clone();
    let neighbors_ca = df
        .column("direct_and_indirect_neighbors")
        .map_err(to_py)?
        .as_materialized_series()
        .list()
        .map_err(to_py)?
        .clone();

    let cluster_ids: Vec<u32> = cluster_ca.into_no_null_iter().collect();
    let mut adjacency: HashMap<u32, HashSet<u32>> = HashMap::new();
    for (i, &cluster) in cluster_ids.iter().enumerate() {
        let neighbors: HashSet<u32> = match neighbors_ca.get_as_series(i) {
            Some(s) => s.u32().map_err(to_py)?.into_no_null_iter().collect(),
            None => HashSet::new(),
        };
        adjacency.insert(cluster, neighbors);
    }

    let color_indices = greedy_color(&cluster_ids, &adjacency);
    let num_colors = color_indices.values().collect::<HashSet<_>>().len();
    let palette = ylorrd_hex_palette(num_colors);

    let mut colors: Vec<String> = Vec::with_capacity(cluster_ids.len());
    let mut darkened: Vec<String> = Vec::with_capacity(cluster_ids.len());
    for &cluster in &cluster_ids {
        let color = palette[color_indices[&cluster] as usize].clone();
        darkened.push(darken_hex_color(&color, 0.5)?);
        colors.push(color);
    }

    let out = DataFrame::new(
        cluster_ids.len(),
        vec![
            UInt32Chunked::from_vec("cluster".into(), cluster_ids).into_column(),
            StringChunked::from_iter_values("color".into(), colors.into_iter()).into_column(),
            StringChunked::from_iter_values("darkened_color".into(), darkened.into_iter())
                .into_column(),
        ],
    )
    .map_err(to_py)?;

    Ok(PyDataFrame(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ylorrd_matches_known_hex_values() {
        // Verified against `sns.color_palette("YlOrRd", n).as_hex()`.
        assert_eq!(ylorrd_hex_palette(2), vec!["#febf5a", "#f43d25"]);
        assert_eq!(ylorrd_hex_palette(3), vec!["#fed976", "#fd8c3c", "#e2191c"]);
    }

    #[test]
    fn greedy_color_avoids_adjacent_same_color() {
        // Triangle: every node needs a distinct color.
        let ids = vec![0, 1, 2];
        let mut adjacency = HashMap::new();
        adjacency.insert(0, HashSet::from([1, 2]));
        adjacency.insert(1, HashSet::from([0, 2]));
        adjacency.insert(2, HashSet::from([0, 1]));
        let colors = greedy_color(&ids, &adjacency);
        assert_ne!(colors[&0], colors[&1]);
        assert_ne!(colors[&1], colors[&2]);
        assert_ne!(colors[&0], colors[&2]);
    }
}
