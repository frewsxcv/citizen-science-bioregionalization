//! Port of `src/dataframes/cluster_boundary.py` (`build_cluster_boundary_df`).

use std::collections::{BTreeMap, HashMap};

use geo::{Polygon, unary_union};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::{to_py, wkb};

/// Build a ClusterBoundarySchema DataFrame: a single boundary geometry per
/// cluster, formed by unioning all its geocodes' hexagon boundaries.
///
/// Mirrors `build_cluster_boundary_df`. Clusters are emitted in ascending
/// order (matching `iter_clusters_and_geocodes`'s `.sort("cluster")`). A
/// single-geocode cluster is encoded as a bare `Polygon`, matching Python
/// (which keeps `cluster_geocode_boundaries[0]` as-is); a multi-geocode
/// cluster is encoded as a `MultiPolygon` (`geo::unary_union`'s return type),
/// even when the pieces fully merge into one connected shape — unlike
/// shapely, which would collapse that case to a bare `Polygon`. This
/// doesn't change the geometry's actual shape (shapely and GEOS both treat a
/// one-polygon `MultiPolygon` as topologically equal to that `Polygon`), only
/// its WKB type tag.
#[pyfunction]
pub fn build_cluster_boundary(
    geocode_cluster_df: PyDataFrame,
    geocode_df: PyDataFrame,
) -> PyResult<PyDataFrame> {
    let geocode_cluster_df: DataFrame = geocode_cluster_df.into();
    let geocode_df: DataFrame = geocode_df.into();

    let geocode_to_boundary: HashMap<u64, Polygon<f64>> = {
        let geocode = geocode_df
            .column("geocode")
            .map_err(to_py)?
            .as_materialized_series()
            .u64()
            .map_err(to_py)?
            .clone();
        let boundary = geocode_df
            .column("boundary")
            .map_err(to_py)?
            .as_materialized_series()
            .binary()
            .map_err(to_py)?
            .clone();
        geocode
            .into_no_null_iter()
            .zip(boundary.iter())
            .map(|(g, b)| {
                (
                    g,
                    wkb::decode_polygon(b.expect("boundary column must be non-null")),
                )
            })
            .collect()
    };

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

    let mut cluster_boundaries: BTreeMap<u32, Vec<Polygon<f64>>> = BTreeMap::new();
    for (cluster, geocode) in cluster_ca
        .into_no_null_iter()
        .zip(geocode_ca.into_no_null_iter())
    {
        if let Some(boundary) = geocode_to_boundary.get(&geocode) {
            cluster_boundaries
                .entry(cluster)
                .or_default()
                .push(boundary.clone());
        }
    }

    let mut clusters: Vec<u32> = Vec::new();
    let mut geometries: Vec<Vec<u8>> = Vec::new();
    for (cluster, boundaries) in cluster_boundaries {
        if boundaries.is_empty() {
            continue;
        }
        let wkb_bytes = if boundaries.len() == 1 {
            wkb::encode_polygon(&boundaries[0])
        } else {
            wkb::encode_multi_polygon(&unary_union(&boundaries))
        };
        clusters.push(cluster);
        geometries.push(wkb_bytes);
    }

    let geometry_ca: BinaryChunked = geometries.iter().map(|v| Some(v.as_slice())).collect();
    let out = DataFrame::new(
        clusters.len(),
        vec![
            UInt32Chunked::from_vec("cluster".into(), clusters).into_column(),
            geometry_ca.with_name("geometry".into()).into_column(),
        ],
    )
    .map_err(to_py)?;

    Ok(PyDataFrame(out))
}
