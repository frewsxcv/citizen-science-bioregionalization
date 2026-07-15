//! Port of `src/dataframes/cluster_neighbors.py` (`build_cluster_neighbors_df`).

use std::collections::{BTreeSet, HashMap, HashSet};

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::to_py;

/// Look up the cluster for a geocode, erroring (mirroring Python's
/// `cluster_for_geocode`, which raises `IndexError` on a missing geocode)
/// rather than silently skipping.
fn cluster_of(map: &HashMap<u64, u32>, geocode: u64) -> PolarsResult<u32> {
    map.get(&geocode).copied().ok_or_else(|| {
        PolarsError::ComputeError(
            format!("geocode {geocode} not found in geocode_cluster_df").into(),
        )
    })
}

/// Build a ClusterNeighborsSchema DataFrame: which clusters are (direct and
/// indirect) neighbors of each other, derived from geocode-level adjacency.
///
/// Mirrors `build_cluster_neighbors_df`.
#[pyfunction]
pub fn build_cluster_neighbors(
    geocode_neighbors_df: PyDataFrame,
    geocode_cluster_df: PyDataFrame,
) -> PyResult<PyDataFrame> {
    let geocode_neighbors_df: DataFrame = geocode_neighbors_df.into();
    let geocode_cluster_df: DataFrame = geocode_cluster_df.into();

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

    let unique_clusters: BTreeSet<u32> = geocode_to_cluster.values().copied().collect();
    let mut direct_neighbors_map: HashMap<u32, HashSet<u32>> = unique_clusters
        .iter()
        .map(|&c| (c, HashSet::new()))
        .collect();
    let mut all_neighbors_map: HashMap<u32, HashSet<u32>> = unique_clusters
        .iter()
        .map(|&c| (c, HashSet::new()))
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

    for i in 0..geocode_neighbors_df.height() {
        let Some(geocode) = geocode_ca.get(i) else {
            continue;
        };
        let current_cluster = cluster_of(&geocode_to_cluster, geocode).map_err(to_py)?;

        if let Some(series) = direct_ca.get_as_series(i) {
            for neighbor in series.u64().map_err(to_py)?.into_no_null_iter() {
                let neighbor_cluster = cluster_of(&geocode_to_cluster, neighbor).map_err(to_py)?;
                if neighbor_cluster != current_cluster {
                    direct_neighbors_map
                        .get_mut(&current_cluster)
                        .expect("cluster set initialized for every unique cluster")
                        .insert(neighbor_cluster);
                    all_neighbors_map
                        .get_mut(&current_cluster)
                        .expect("cluster set initialized for every unique cluster")
                        .insert(neighbor_cluster);
                }
            }
        }

        if let Some(series) = direct_and_indirect_ca.get_as_series(i) {
            for neighbor in series.u64().map_err(to_py)?.into_no_null_iter() {
                let neighbor_cluster = cluster_of(&geocode_to_cluster, neighbor).map_err(to_py)?;
                if neighbor_cluster != current_cluster {
                    all_neighbors_map
                        .get_mut(&current_cluster)
                        .expect("cluster set initialized for every unique cluster")
                        .insert(neighbor_cluster);
                }
            }
        }
    }

    let n = unique_clusters.len();
    let cluster_col =
        UInt32Chunked::from_vec("cluster".into(), unique_clusters.iter().copied().collect())
            .into_column();

    let mut direct_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        "direct_neighbors".into(),
        n,
        n,
        DataType::UInt32,
    );
    let mut direct_and_indirect_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        "direct_and_indirect_neighbors".into(),
        n,
        n,
        DataType::UInt32,
    );
    for &cluster in &unique_clusters {
        let direct: Vec<u32> = direct_neighbors_map[&cluster].iter().copied().collect();
        let all: Vec<u32> = all_neighbors_map[&cluster].iter().copied().collect();
        direct_builder.append_slice(&direct);
        direct_and_indirect_builder.append_slice(&all);
    }

    let out = DataFrame::new(
        n,
        vec![
            cluster_col,
            direct_builder.finish().into_series().into_column(),
            direct_and_indirect_builder
                .finish()
                .into_series()
                .into_column(),
        ],
    )
    .map_err(to_py)?;

    Ok(PyDataFrame(out))
}
