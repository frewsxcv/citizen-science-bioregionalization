//! Port of `src/dataframes/geocode_neighbors.py`.

use std::collections::{HashMap, HashSet};

use h3o::CellIndex;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::dataframes::taxonomy::known_geocodes;
use crate::{to_py, wkb};

const MAX_NUM_NEIGHBORS: usize = 6;

/// Working state for one geocode row while building/reducing the neighbor graph.
struct Node {
    geocode: u64,
    center: (f64, f64),
    direct_neighbors: Vec<u64>,
    direct_and_indirect_neighbors: Vec<u64>,
}

/// Minimal union-find over row indices, used to track connected components
/// without pulling in a graph crate.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, i: usize) -> usize {
        if self.parent[i] != i {
            self.parent[i] = self.find(self.parent[i]);
        }
        self.parent[i]
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }

    fn num_components(&mut self) -> usize {
        let n = self.parent.len();
        let roots: HashSet<usize> = (0..n).map(|i| self.find(i)).collect();
        roots.len()
    }
}

/// Repeatedly connect the first (by row order) connected component to the
/// spatially nearest node in another component (skipping nodes that already
/// have `MAX_NUM_NEIGHBORS` direct neighbors), until a single component
/// remains. Mirrors `_reduce_connected_components_to_one`.
///
/// Distance ties (equidistant candidate pairs) are broken by row order here,
/// which is not guaranteed to match shapely/GEOS's internal tie-break in the
/// Python implementation; in practice centers are real-valued coordinates so
/// exact ties don't occur.
fn reduce_connected_components_to_one(nodes: &mut [Node]) -> PolarsResult<()> {
    let n = nodes.len();
    if n == 0 {
        return Ok(());
    }
    let index_of: HashMap<u64, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.geocode, i))
        .collect();

    let mut uf = UnionFind::new(n);
    for (i, node) in nodes.iter().enumerate() {
        for &neighbor in &node.direct_neighbors {
            if let Some(&j) = index_of.get(&neighbor) {
                uf.union(i, j);
            }
        }
    }

    while uf.num_components() > 1 {
        let first_root = uf.find(0);

        let mut best: Option<(usize, usize, f64)> = None;
        for i in 0..n {
            if uf.find(i) != first_root {
                continue;
            }
            for (j, other) in nodes.iter().enumerate() {
                if uf.find(j) == first_root || other.direct_neighbors.len() == MAX_NUM_NEIGHBORS {
                    continue;
                }
                let (xi, yi) = nodes[i].center;
                let (xj, yj) = other.center;
                let dist_sq = (xi - xj).powi(2) + (yi - yj).powi(2);
                if best.is_none_or(|(_, _, best_dist)| dist_sq < best_dist) {
                    best = Some((i, j, dist_sq));
                }
            }
        }

        let (i, j, _) = best.ok_or_else(|| {
            PolarsError::ComputeError(
                "No closest pair found while connecting geocode neighbor components".into(),
            )
        })?;

        uf.union(i, j);
        let (geocode_i, geocode_j) = (nodes[i].geocode, nodes[j].geocode);
        nodes[i].direct_and_indirect_neighbors.push(geocode_j);
        nodes[j].direct_and_indirect_neighbors.push(geocode_i);
    }

    Ok(())
}

/// Decode the `geocode` (UInt64) and `center` (WKB Point Binary) columns of a
/// DataFrame into row-aligned vectors.
fn geocode_and_centers(df: &DataFrame) -> PolarsResult<(Vec<u64>, Vec<(f64, f64)>)> {
    let geocodes: Vec<u64> = df
        .column("geocode")?
        .as_materialized_series()
        .u64()?
        .into_no_null_iter()
        .collect();
    let centers: Vec<(f64, f64)> = df
        .column("center")?
        .as_materialized_series()
        .binary()?
        .iter()
        .map(|b| wkb::decode_point(b.expect("center column must be non-null")))
        .collect();
    Ok((geocodes, centers))
}

/// Build a DataFrame's `geocode` (UInt64) + two List<UInt64> neighbor columns.
fn neighbors_to_df(nodes: &[Node]) -> PolarsResult<DataFrame> {
    let geocode_col =
        UInt64Chunked::from_vec("geocode".into(), nodes.iter().map(|n| n.geocode).collect())
            .into_column();

    let mut direct_builder = ListPrimitiveChunkedBuilder::<UInt64Type>::new(
        "direct_neighbors".into(),
        nodes.len(),
        nodes.len() * MAX_NUM_NEIGHBORS,
        DataType::UInt64,
    );
    let mut direct_and_indirect_builder = ListPrimitiveChunkedBuilder::<UInt64Type>::new(
        "direct_and_indirect_neighbors".into(),
        nodes.len(),
        nodes.len() * MAX_NUM_NEIGHBORS,
        DataType::UInt64,
    );
    for node in nodes {
        direct_builder.append_slice(&node.direct_neighbors);
        direct_and_indirect_builder.append_slice(&node.direct_and_indirect_neighbors);
    }

    DataFrame::new(
        nodes.len(),
        vec![
            geocode_col,
            direct_builder.finish().into_series().into_column(),
            direct_and_indirect_builder
                .finish()
                .into_series()
                .into_column(),
        ],
    )
}

/// Build a GeocodeNeighborsSchema DataFrame from geocode spatial data.
///
/// Mirrors `src/dataframes/geocode_neighbors.py::build_geocode_neighbors_df`:
/// H3 grid-ring adjacency restricted to this DataFrame's own geocode set, then
/// indirect edges added (nearest-point heuristic) until the neighbor graph is
/// a single connected component.
#[pyfunction]
pub fn build_geocode_neighbors(geocode_df: PyDataFrame) -> PyResult<PyDataFrame> {
    let geocode_df: DataFrame = geocode_df.into();

    let (geocodes, centers) = geocode_and_centers(&geocode_df).map_err(to_py)?;
    let known: HashSet<u64> = geocodes.iter().copied().collect();

    let mut nodes: Vec<Node> = Vec::with_capacity(geocodes.len());
    for (&geocode, &center) in geocodes.iter().zip(centers.iter()) {
        let cell = CellIndex::try_from(geocode).map_err(|e| {
            to_py(PolarsError::ComputeError(
                format!("invalid H3 cell {geocode}: {e}").into(),
            ))
        })?;
        let direct_neighbors: Vec<u64> = cell
            .grid_ring_fast(1)
            .flatten()
            .map(u64::from)
            .filter(|c| known.contains(c))
            .collect();
        nodes.push(Node {
            geocode,
            center,
            direct_and_indirect_neighbors: direct_neighbors.clone(),
            direct_neighbors,
        });
    }

    reduce_connected_components_to_one(&mut nodes).map_err(to_py)?;

    let out = neighbors_to_df(&nodes).map_err(to_py)?;
    Ok(PyDataFrame(out))
}

/// Build a GeocodeNeighborsSchema DataFrame restricted to non-edge geocodes.
///
/// Mirrors `build_geocode_neighbors_no_edges_df`: filter neighbor lists down
/// to the valid (non-edge) geocode set, re-attach centers, and re-run the
/// connectivity fixup since removing edge geocodes can re-fragment the graph.
#[pyfunction]
pub fn build_geocode_neighbors_no_edges(
    geocode_neighbors_df: PyDataFrame,
    geocode_no_edges_df: PyDataFrame,
) -> PyResult<PyDataFrame> {
    let geocode_neighbors_df: DataFrame = geocode_neighbors_df.into();
    let geocode_no_edges_df: DataFrame = geocode_no_edges_df.into();

    let valid = known_geocodes(&geocode_no_edges_df).map_err(to_py)?;
    let (no_edges_geocodes, no_edges_centers) =
        geocode_and_centers(&geocode_no_edges_df).map_err(to_py)?;
    let center_by_geocode: HashMap<u64, (f64, f64)> = no_edges_geocodes
        .into_iter()
        .zip(no_edges_centers)
        .collect();

    let geocode_ca = geocode_neighbors_df
        .column("geocode")
        .map_err(to_py)?
        .as_materialized_series()
        .u64()
        .map_err(to_py)?
        .clone();
    let direct_ca = geocode_neighbors_df
        .column("direct_neighbors")
        .map_err(to_py)?
        .as_materialized_series()
        .list()
        .map_err(to_py)?
        .clone();
    let direct_and_indirect_ca = geocode_neighbors_df
        .column("direct_and_indirect_neighbors")
        .map_err(to_py)?
        .as_materialized_series()
        .list()
        .map_err(to_py)?
        .clone();

    let mut nodes: Vec<Node> = Vec::new();
    for i in 0..geocode_neighbors_df.height() {
        let Some(geocode) = geocode_ca.get(i) else {
            continue;
        };
        if !valid.contains(&geocode) {
            continue;
        }
        let list_as_u64 = |list_ca: &ListChunked| -> PolarsResult<Vec<u64>> {
            Ok(list_ca
                .get_as_series(i)
                .map(|s| {
                    s.u64()
                        .expect("neighbor list column must be UInt64")
                        .into_no_null_iter()
                        .filter(|c| valid.contains(c))
                        .collect()
                })
                .unwrap_or_default())
        };
        let direct_neighbors = list_as_u64(&direct_ca).map_err(to_py)?;
        let direct_and_indirect_neighbors = list_as_u64(&direct_and_indirect_ca).map_err(to_py)?;
        let center = *center_by_geocode.get(&geocode).ok_or_else(|| {
            to_py(PolarsError::ComputeError(
                format!("geocode {geocode} missing from geocode_no_edges_df").into(),
            ))
        })?;
        nodes.push(Node {
            geocode,
            center,
            direct_neighbors,
            direct_and_indirect_neighbors,
        });
    }

    reduce_connected_components_to_one(&mut nodes).map_err(to_py)?;
    nodes.sort_by_key(|n| n.geocode);

    let out = neighbors_to_df(&nodes).map_err(to_py)?;
    Ok(PyDataFrame(out))
}
