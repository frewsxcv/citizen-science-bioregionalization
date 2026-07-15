//! Port of `src/matrices/geocode_connectivity.py`
//! (`GeocodeConnectivityMatrix.build`).

use std::collections::HashMap;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::to_py;

/// BFS reachability check: is `matrix` (as an undirected adjacency matrix) a
/// single connected component?
fn is_single_connected_component(matrix: &[Vec<i64>]) -> bool {
    let n = matrix.len();
    if n == 0 {
        return true;
    }
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    visited[0] = true;
    let mut count = 1;
    while let Some(u) = stack.pop() {
        for (v, &connected) in matrix[u].iter().enumerate() {
            if connected != 0 && !visited[v] {
                visited[v] = true;
                count += 1;
                stack.push(v);
            }
        }
    }
    count == n
}

/// Build a dense (num_geocodes x num_geocodes) 0/1 adjacency matrix from a
/// GeocodeNeighborsSchema DataFrame's `direct_and_indirect_neighbors` column.
///
/// Mirrors `GeocodeConnectivityMatrix.build`: returned as a nested list (PyO3
/// converts `Vec<Vec<i64>>` to a Python list of lists — note `Vec<u8>` would
/// convert to Python `bytes` instead, which is why cells are `i64` rather than
/// a narrower unsigned type), which the Python side wraps with `np.array(...)`
/// to match the existing `GeocodeConnectivityMatrix` type (`dtype=int`, i.e.
/// int64). Errors (instead of Python's `assert`) if the graph isn't a single
/// connected component, since that invariant is expected to already hold by
/// construction from `geocode_neighbors`.
#[pyfunction]
pub fn build_geocode_connectivity_matrix(
    geocode_neighbors_df: PyDataFrame,
) -> PyResult<Vec<Vec<i64>>> {
    let df: DataFrame = geocode_neighbors_df.into();
    let n = df.height();

    let geocode_ca = df
        .column("geocode")
        .map_err(to_py)?
        .as_materialized_series()
        .u64()
        .map_err(to_py)?
        .clone();
    let neighbors_ca = df
        .column("direct_and_indirect_neighbors")
        .map_err(to_py)?
        .as_materialized_series()
        .list()
        .map_err(to_py)?
        .clone();

    let geocode_to_index: HashMap<u64, usize> = geocode_ca
        .iter()
        .enumerate()
        .filter_map(|(i, g)| g.map(|g| (g, i)))
        .collect();

    let mut matrix = vec![vec![0i64; n]; n];
    for i in 0..n {
        let Some(series) = neighbors_ca.get_as_series(i) else {
            continue;
        };
        let neighbors = series.u64().map_err(to_py)?;
        for neighbor in neighbors.into_no_null_iter() {
            if let Some(&j) = geocode_to_index.get(&neighbor) {
                matrix[i][j] = 1;
            }
        }
    }

    if !is_single_connected_component(&matrix) {
        return Err(PyValueError::new_err(
            "geocode connectivity matrix is not a single connected component",
        ));
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_disconnected_matrix() {
        let matrix = vec![vec![0, 1, 0], vec![1, 0, 0], vec![0, 0, 0]];
        assert!(!is_single_connected_component(&matrix));
    }

    #[test]
    fn detects_connected_matrix() {
        let matrix = vec![vec![0, 1, 0], vec![1, 0, 1], vec![0, 1, 0]];
        assert!(is_single_connected_component(&matrix));
    }
}
