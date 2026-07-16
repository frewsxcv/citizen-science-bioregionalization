//! Port of `src/geojson.py` (`build_geojson_feature_collection`).
//!
//! `write_geojson` (a plain file write) stays in Python — it's I/O, not
//! compute, and this crate hasn't cut over any pipeline call sites yet (see
//! `RUST_MIGRATION_PLAN.md`). This function returns the FeatureCollection as
//! a JSON string rather than a `geojson.FeatureCollection` Python object: no
//! code in the codebase does an `isinstance` check against that type (it's
//! only ever passed straight through to `geojson.dump`, which just needs
//! something JSON-serializable), so a plain JSON string is a safe stand-in
//! for verifying this function's logic, without adding a JSON-building
//! dependency for this crate's one and only user of one.

use geo::{Coord, LineString, Polygon};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

use crate::to_py;
use crate::wkb;

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn coord_to_json(c: &Coord<f64>) -> String {
    format!("[{},{}]", c.x, c.y)
}

fn ring_to_json(ring: &LineString<f64>) -> String {
    let coords: Vec<String> = ring.coords().map(coord_to_json).collect();
    format!("[{}]", coords.join(","))
}

/// A polygon's GeoJSON `coordinates` value: an array of rings (exterior
/// first, then any holes).
fn polygon_rings_to_json(poly: &Polygon<f64>) -> String {
    let mut rings = vec![ring_to_json(poly.exterior())];
    rings.extend(poly.interiors().iter().map(ring_to_json));
    format!("[{}]", rings.join(","))
}

/// A GeoJSON `geometry` object (`{"type": ..., "coordinates": ...}`),
/// matching `shapely.geometry.mapping(geom)`'s output shape for a Polygon or
/// MultiPolygon.
fn geometry_to_json(geom: &wkb::Geometry) -> String {
    match geom {
        wkb::Geometry::Polygon(poly) => {
            format!(
                r#"{{"type":"Polygon","coordinates":{}}}"#,
                polygon_rings_to_json(poly)
            )
        }
        wkb::Geometry::MultiPolygon(mp) => {
            let polys: Vec<String> = mp.0.iter().map(polygon_rings_to_json).collect();
            format!(
                r#"{{"type":"MultiPolygon","coordinates":[{}]}}"#,
                polys.join(",")
            )
        }
    }
}

/// A single GeoJSON Feature, matching `build_geojson_feature`'s properties
/// and geometry exactly.
fn feature_to_json(
    cluster: u32,
    color: &str,
    darkened_color: &str,
    geom: &wkb::Geometry,
) -> String {
    format!(
        r#"{{"type":"Feature","properties":{{"color":{},"fillColor":{},"fillOpacity":0.7,"weight":1,"cluster":{}}},"geometry":{}}}"#,
        json_escape(darkened_color),
        json_escape(color),
        cluster,
        geometry_to_json(geom),
    )
}

/// Build a GeoJSON FeatureCollection (as a JSON string) from cluster
/// boundaries and colors, joined on `cluster` (inner join, matching Python's
/// default `.join()`).
///
/// Mirrors `build_geojson_feature_collection`.
#[pyfunction]
pub fn build_geojson_feature_collection(
    cluster_boundary_df: PyDataFrame,
    cluster_colors_df: PyDataFrame,
) -> PyResult<String> {
    let cluster_boundary_df: DataFrame = cluster_boundary_df.into();
    let cluster_colors_df: DataFrame = cluster_colors_df.into();

    let colors: HashMap<u32, (String, String)> = {
        let cluster_ca = cluster_colors_df
            .column("cluster")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let color_ca = cluster_colors_df
            .column("color")
            .map_err(to_py)?
            .as_materialized_series()
            .str()
            .map_err(to_py)?
            .clone();
        let darkened_ca = cluster_colors_df
            .column("darkened_color")
            .map_err(to_py)?
            .as_materialized_series()
            .str()
            .map_err(to_py)?
            .clone();
        cluster_ca
            .into_no_null_iter()
            .zip(color_ca.iter())
            .zip(darkened_ca.iter())
            .map(|((c, color), darkened)| {
                (
                    c,
                    (
                        color.expect("color column must be non-null").to_string(),
                        darkened
                            .expect("darkened_color column must be non-null")
                            .to_string(),
                    ),
                )
            })
            .collect()
    };

    let cluster_ca = cluster_boundary_df
        .column("cluster")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .clone();
    let geometry_ca = cluster_boundary_df
        .column("geometry")
        .map_err(to_py)?
        .as_materialized_series()
        .binary()
        .map_err(to_py)?
        .clone();

    let mut features: Vec<String> = Vec::new();
    for (cluster, geometry_bytes) in cluster_ca.into_no_null_iter().zip(geometry_ca.iter()) {
        let Some((color, darkened_color)) = colors.get(&cluster) else {
            continue; // inner join: skip clusters with no matching color row
        };
        let geom = wkb::decode_geometry(geometry_bytes.expect("geometry column must be non-null"));
        features.push(feature_to_json(cluster, color, darkened_color, &geom));
    }

    Ok(format!(
        r#"{{"type":"FeatureCollection","features":[{}]}}"#,
        features.join(",")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_json_matches_expected_shape() {
        // build_geojson_feature maps properties["color"] = darkened_color,
        // properties["fillColor"] = color -- verify we didn't swap them.
        let ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
        let poly = crate::wkb::decode_polygon(&crate::wkb::polygon(&ring));
        let json = feature_to_json(3, "#abcdef", "#556677", &wkb::Geometry::Polygon(poly));
        assert!(json.contains(r#""type":"Feature""#));
        assert!(json.contains(r#""cluster":3"#));
        assert!(json.contains("\"color\":\"#556677\""));
        assert!(json.contains("\"fillColor\":\"#abcdef\""));
        assert!(json.contains(r#""type":"Polygon""#));
        assert!(json.contains("\"coordinates\":[[[0,0],[1,0],[1,1],[0,0]]]"));
    }

    #[test]
    fn json_escape_handles_quotes_and_backslashes() {
        assert_eq!(json_escape("a\"b\\c"), "\"a\\\"b\\\\c\"");
    }
}
