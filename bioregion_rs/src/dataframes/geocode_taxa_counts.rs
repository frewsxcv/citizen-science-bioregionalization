//! Port of `src/dataframes/geocode_taxa_counts.py` (`build_geocode_taxa_counts_lf`).

use std::collections::HashMap;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::dataframes::taxonomy::known_geocodes;
use crate::geocode::{filter_by_bounding_box_df, with_geocode_df};
use crate::to_py;

/// Map (scientificName, gbifTaxonId) -> taxonId from a TaxonomySchema DataFrame.
fn taxon_id_lookup(taxonomy_df: &DataFrame) -> PolarsResult<HashMap<(Option<String>, u32), u32>> {
    let scientific_name = taxonomy_df
        .column("scientificName")?
        .as_materialized_series()
        .str()?
        .clone();
    let gbif_taxon_id = taxonomy_df
        .column("gbifTaxonId")?
        .as_materialized_series()
        .u32()?
        .clone();
    let taxon_id = taxonomy_df
        .column("taxonId")?
        .as_materialized_series()
        .u32()?
        .clone();

    Ok(scientific_name
        .iter()
        .zip(gbif_taxon_id.iter())
        .zip(taxon_id.iter())
        .filter_map(|((name, gbif_id), tid)| {
            let gbif_id = gbif_id?;
            let tid = tid?;
            Some(((name.map(str::to_string), gbif_id), tid))
        })
        .collect())
}

/// Build a GeocodeTaxaCountsSchema DataFrame from Darwin Core occurrence data.
///
/// Mirrors `src/dataframes/geocode_taxa_counts.py::build_geocode_taxa_counts_lf`:
/// joins occurrences (filtered to known geocodes) against the taxonomy table to
/// resolve `taxonId`, then aggregates occurrence counts per (geocode, taxonId),
/// treating a null `individualCount` as 1.
#[pyfunction]
#[pyo3(signature = (darwin_core_df, geocode_precision, taxonomy_df, geocode_df, min_lat, max_lat, min_lng, max_lng))]
#[allow(clippy::too_many_arguments)]
pub fn build_geocode_taxa_counts(
    darwin_core_df: PyDataFrame,
    geocode_precision: u8,
    taxonomy_df: PyDataFrame,
    geocode_df: PyDataFrame,
    min_lat: f64,
    max_lat: f64,
    min_lng: f64,
    max_lng: f64,
) -> PyResult<PyDataFrame> {
    let darwin_core_df: DataFrame = darwin_core_df.into();
    let taxonomy_df: DataFrame = taxonomy_df.into();
    let geocode_df: DataFrame = geocode_df.into();

    let mut df = darwin_core_df
        .select([
            "decimalLatitude",
            "decimalLongitude",
            "scientificName",
            "taxonKey",
            "individualCount",
        ])
        .map_err(to_py)?;
    df.rename("taxonKey", "gbifTaxonId".into()).map_err(to_py)?;

    let df = filter_by_bounding_box_df(
        &df,
        min_lat,
        max_lat,
        min_lng,
        max_lng,
        "decimalLatitude",
        "decimalLongitude",
    )
    .map_err(to_py)?;
    let df = with_geocode_df(&df, geocode_precision).map_err(to_py)?;

    let known = known_geocodes(&geocode_df).map_err(to_py)?;
    let geocode_col = df
        .column("geocode")
        .map_err(to_py)?
        .as_materialized_series()
        .u64()
        .map_err(to_py)?;
    let mask: BooleanChunked = geocode_col
        .iter()
        .map(|g| g.is_some_and(|g| known.contains(&g)))
        .collect();
    let df = df.filter(&mask).map_err(to_py)?;

    let lookup = taxon_id_lookup(&taxonomy_df).map_err(to_py)?;

    let geocode_ca = df
        .column("geocode")
        .map_err(to_py)?
        .as_materialized_series()
        .u64()
        .map_err(to_py)?
        .clone();
    let name_ca = df
        .column("scientificName")
        .map_err(to_py)?
        .as_materialized_series()
        .str()
        .map_err(to_py)?
        .clone();
    let gbif_ca = df
        .column("gbifTaxonId")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .clone();
    let count_ca = df
        .column("individualCount")
        .map_err(to_py)?
        .as_materialized_series()
        .i32()
        .map_err(to_py)?
        .clone();

    // Inner join against the taxonomy lookup, then fill a null individualCount
    // with 1 (mirrors `.fill_null(1)` before the sum aggregation).
    let mut geocodes: Vec<u64> = Vec::new();
    let mut taxon_ids: Vec<u32> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    for (((geocode, name), gbif_id), count) in geocode_ca
        .iter()
        .zip(name_ca.iter())
        .zip(gbif_ca.iter())
        .zip(count_ca.iter())
    {
        let (Some(geocode), Some(gbif_id)) = (geocode, gbif_id) else {
            continue;
        };
        let key = (name.map(str::to_string), gbif_id);
        if let Some(&taxon_id) = lookup.get(&key) {
            geocodes.push(geocode);
            taxon_ids.push(taxon_id);
            counts.push(count.map(|c| c as u32).unwrap_or(1));
        }
    }

    let joined = DataFrame::new(
        geocodes.len(),
        vec![
            UInt64Chunked::from_vec("geocode".into(), geocodes).into_column(),
            UInt32Chunked::from_vec("taxonId".into(), taxon_ids).into_column(),
            UInt32Chunked::from_vec("individualCount".into(), counts).into_column(),
        ],
    )
    .map_err(to_py)?;

    // `GroupBy::sum` is deprecated in favor of lazy `.agg()` expressions, which
    // aren't available here (the `lazy` feature doesn't compile on this
    // toolchain — see the crate README). This eager aggregation is correct and
    // is the only group-by-sum path available without it.
    #[allow(deprecated)]
    let grouped = joined
        .group_by(["geocode", "taxonId"])
        .map_err(to_py)?
        .select(["individualCount"])
        .sum()
        .map_err(to_py)?;

    let mut grouped = grouped;
    grouped
        .rename("individualCount_sum", "count".into())
        .map_err(to_py)?;

    let out = grouped
        .sort(["geocode"], SortMultipleOptions::default())
        .map_err(to_py)?;

    Ok(PyDataFrame(out))
}
