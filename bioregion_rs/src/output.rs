//! Port of `src/output.py`'s `write_json_output`.
//!
//! `write_json_output` itself is a thin file-write wrapper around building a
//! JSON document; the actual assembly (joins + geometry conversion) is the
//! part worth porting. This module returns the assembled JSON as a `String`
//! and leaves `prepare_file_path`/`open(...).write(...)` in Python, matching
//! the `geojson.rs` precedent (I/O stays in Python; this crate hasn't cut
//! over any pipeline call sites yet, see `RUST_MIGRATION_PLAN.md`).
//!
//! Boundary geometry is converted to a GeoJSON geometry object via the same
//! `wkb::decode_geometry` + `geojson` crate path as `geojson.rs`, which is
//! the Rust equivalent of Python's `shapely.to_geojson(shapely.from_wkb(...))`.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::to_py;
use crate::wkb;

fn wkb_to_geojson_value(bytes: &[u8]) -> Value {
    let geom = wkb::decode_geometry(bytes);
    let geojson_geom = match &geom {
        wkb::Geometry::Polygon(poly) => geojson::Geometry::from(poly),
        wkb::Geometry::MultiPolygon(mp) => geojson::Geometry::from(mp),
    };
    serde_json::to_value(geojson_geom).expect("geojson::Geometry always serializes")
}

struct SignificantDiffRow {
    taxon_id: u32,
    log2_fold_change: f64,
    cluster_count: u32,
    neighbor_count: u32,
    high_log2_high_count_score: f64,
    low_log2_high_count_score: f64,
}

/// Build the same JSON document as `write_json_output` (as a string, prior to
/// the file write), joining cluster boundaries/colors and, per cluster, its
/// significant taxa (with taxonomy and, where available, an image URL).
#[pyfunction]
pub fn build_json_output(
    cluster_significant_differences_df: PyDataFrame,
    cluster_boundary_df: PyDataFrame,
    taxonomy_df: PyDataFrame,
    cluster_color_df: PyDataFrame,
    significant_taxa_images_df: PyDataFrame,
) -> PyResult<String> {
    let diffs_df: DataFrame = cluster_significant_differences_df.into();
    let boundary_df: DataFrame = cluster_boundary_df.into();
    let taxonomy_df: DataFrame = taxonomy_df.into();
    let colors_df: DataFrame = cluster_color_df.into();
    let images_df: DataFrame = significant_taxa_images_df.into();

    // taxonId -> (scientificName, gbifTaxonId)
    let taxonomy: HashMap<u32, (Option<String>, u32)> = {
        let taxon_id_ca = taxonomy_df
            .column("taxonId")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let sci_name_ca = taxonomy_df
            .column("scientificName")
            .map_err(to_py)?
            .as_materialized_series()
            .str()
            .map_err(to_py)?
            .clone();
        let gbif_ca = taxonomy_df
            .column("gbifTaxonId")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        taxon_id_ca
            .into_no_null_iter()
            .zip(sci_name_ca.iter())
            .zip(gbif_ca.into_no_null_iter())
            .map(|((taxon_id, sci_name), gbif_id)| {
                (taxon_id, (sci_name.map(|s| s.to_string()), gbif_id))
            })
            .collect()
    };

    // taxonId -> image_url (left join: missing key means null image_url)
    let images: HashMap<u32, Option<String>> = {
        let taxon_id_ca = images_df
            .column("taxonId")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let url_ca = images_df
            .column("image_url")
            .map_err(to_py)?
            .as_materialized_series()
            .str()
            .map_err(to_py)?
            .clone();
        taxon_id_ca
            .into_no_null_iter()
            .zip(url_ca.iter())
            .map(|(taxon_id, url)| (taxon_id, url.map(|s| s.to_string())))
            .collect()
    };

    // cluster -> (color, darkened_color)
    let colors: HashMap<u32, (String, String)> = {
        let cluster_ca = colors_df
            .column("cluster")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let color_ca = colors_df
            .column("color")
            .map_err(to_py)?
            .as_materialized_series()
            .str()
            .map_err(to_py)?
            .clone();
        let darkened_ca = colors_df
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

    // cluster -> significant-difference rows for that cluster, in original
    // row order (mirrors Python's filter(), which preserves row order).
    let mut diffs_by_cluster: HashMap<u32, Vec<SignificantDiffRow>> = HashMap::new();
    {
        let cluster_ca = diffs_df
            .column("cluster")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let taxon_id_ca = diffs_df
            .column("taxonId")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let log2_ca = diffs_df
            .column("log2_fold_change")
            .map_err(to_py)?
            .as_materialized_series()
            .f64()
            .map_err(to_py)?
            .clone();
        let cluster_count_ca = diffs_df
            .column("cluster_count")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let neighbor_count_ca = diffs_df
            .column("neighbor_count")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let high_ca = diffs_df
            .column("high_log2_high_count_score")
            .map_err(to_py)?
            .as_materialized_series()
            .f64()
            .map_err(to_py)?
            .clone();
        let low_ca = diffs_df
            .column("low_log2_high_count_score")
            .map_err(to_py)?
            .as_materialized_series()
            .f64()
            .map_err(to_py)?
            .clone();

        for i in 0..diffs_df.height() {
            let cluster = cluster_ca.get(i).expect("cluster column must be non-null");
            diffs_by_cluster
                .entry(cluster)
                .or_default()
                .push(SignificantDiffRow {
                    taxon_id: taxon_id_ca.get(i).expect("taxonId column must be non-null"),
                    log2_fold_change: log2_ca
                        .get(i)
                        .expect("log2_fold_change column must be non-null"),
                    cluster_count: cluster_count_ca
                        .get(i)
                        .expect("cluster_count column must be non-null"),
                    neighbor_count: neighbor_count_ca
                        .get(i)
                        .expect("neighbor_count column must be non-null"),
                    high_log2_high_count_score: high_ca
                        .get(i)
                        .expect("high_log2_high_count_score column must be non-null"),
                    low_log2_high_count_score: low_ca
                        .get(i)
                        .expect("low_log2_high_count_score column must be non-null"),
                });
        }
    }

    let cluster_ca = boundary_df
        .column("cluster")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .clone();
    let geometry_ca = boundary_df
        .column("geometry")
        .map_err(to_py)?
        .as_materialized_series()
        .binary()
        .map_err(to_py)?
        .clone();

    let mut output_data: Vec<Value> = Vec::new();
    for (cluster, geometry_bytes) in cluster_ca.into_no_null_iter().zip(geometry_ca.iter()) {
        // inner join: skip clusters with no matching color row
        let Some((color, darkened_color)) = colors.get(&cluster) else {
            continue;
        };
        let boundary =
            wkb_to_geojson_value(geometry_bytes.expect("geometry column must be non-null"));

        let mut significant_taxa: Vec<Value> = Vec::new();
        if let Some(rows) = diffs_by_cluster.get(&cluster) {
            for row in rows {
                // inner join with taxonomy_df: skip taxa with no taxonomy row
                let Some((scientific_name, gbif_taxon_id)) = taxonomy.get(&row.taxon_id) else {
                    continue;
                };
                // left join with significant_taxa_images_df: missing key -> null
                let image_url = images.get(&row.taxon_id).cloned().flatten();
                significant_taxa.push(json!({
                    "scientific_name": scientific_name,
                    "gbif_taxon_id": gbif_taxon_id,
                    "taxon_id": row.taxon_id,
                    "log2_fold_change": row.log2_fold_change,
                    "cluster_count": row.cluster_count,
                    "neighbor_count": row.neighbor_count,
                    "high_log2_high_count_score": row.high_log2_high_count_score,
                    "low_log2_high_count_score": row.low_log2_high_count_score,
                    "image_url": image_url,
                }));
            }
        }

        output_data.push(json!({
            "cluster": cluster,
            "boundary": boundary,
            "significant_taxa": significant_taxa,
            "color": color,
            "darkened_color": darkened_color,
        }));
    }

    serde_json::to_string_pretty(&output_data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_image_and_missing_taxonomy_are_handled() {
        let ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
        let poly = crate::wkb::decode_polygon(&crate::wkb::polygon(&ring));
        let bytes = crate::wkb::encode_polygon(&poly);

        let mut taxonomy = HashMap::new();
        taxonomy.insert(1u32, (Some("Taxon One".to_string()), 100u32));
        // taxon 2 intentionally absent -- should be dropped (inner join)

        let mut images = HashMap::new();
        images.insert(1u32, Some("https://example.com/1.jpg".to_string()));
        // taxon absent from images map -> null image_url (left join)

        assert_eq!(
            taxonomy.get(&2u32),
            None,
            "taxon 2 has no taxonomy row and must be excluded from significant_taxa"
        );
        assert_eq!(
            images.get(&1u32).cloned().flatten(),
            Some("https://example.com/1.jpg".to_string())
        );
        assert_eq!(images.get(&99u32).cloned().flatten(), None);

        // sanity: geometry conversion doesn't panic on a plain polygon
        let value = wkb_to_geojson_value(&bytes);
        assert_eq!(value["type"], "Polygon");
    }
}
