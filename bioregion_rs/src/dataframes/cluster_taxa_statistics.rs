//! Port of `src/dataframes/cluster_taxa_statistics.py`
//! (`build_cluster_taxa_statistics_df`).

use std::collections::HashMap;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::to_py;

/// Column-major (geocode, taxonId, count) view of a GeocodeTaxaCountsSchema
/// DataFrame.
fn geocode_taxon_counts(df: &DataFrame) -> PolarsResult<Vec<(u64, u32, u32)>> {
    let geocode = df
        .column("geocode")?
        .as_materialized_series()
        .u64()?
        .clone();
    let taxon_id = df
        .column("taxonId")?
        .as_materialized_series()
        .u32()?
        .clone();
    let count = df.column("count")?.as_materialized_series().u32()?.clone();
    Ok(geocode
        .into_no_null_iter()
        .zip(taxon_id.into_no_null_iter())
        .zip(count.into_no_null_iter())
        .map(|((g, t), c)| (g, t, c))
        .collect())
}

/// Build a ClusterTaxaStatisticsSchema DataFrame: per-taxon (count, average)
/// stats overall (`cluster = null`) and per cluster.
///
/// Mirrors `build_cluster_taxa_statistics_df`. The join against `taxonomy_df`
/// (on `taxonId`) only ever narrows `geocode_taxa_counts_df` to rows whose
/// `taxonId` taxonomy_df actually has (taxonId is a unique key there), so it's
/// implemented as a semi-filter rather than a real join — no taxonomy column
/// is otherwise used downstream.
#[pyfunction]
pub fn build_cluster_taxa_statistics(
    geocode_taxa_counts_df: PyDataFrame,
    geocode_cluster_df: PyDataFrame,
    taxonomy_df: PyDataFrame,
) -> PyResult<PyDataFrame> {
    let geocode_taxa_counts_df: DataFrame = geocode_taxa_counts_df.into();
    let geocode_cluster_df: DataFrame = geocode_cluster_df.into();
    let taxonomy_df: DataFrame = taxonomy_df.into();

    let taxonomy_taxon_ids: std::collections::HashSet<u32> = taxonomy_df
        .column("taxonId")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .into_no_null_iter()
        .collect();

    let joined: Vec<(u64, u32, u32)> = geocode_taxon_counts(&geocode_taxa_counts_df)
        .map_err(to_py)?
        .into_iter()
        .filter(|&(_, taxon_id, _)| taxonomy_taxon_ids.contains(&taxon_id))
        .collect();

    let total_count: u64 = joined.iter().map(|&(_, _, count)| u64::from(count)).sum();

    // Overall stats (cluster = null): sum(count) and average per taxonId.
    let mut overall_counts: HashMap<u32, u64> = HashMap::new();
    for &(_, taxon_id, count) in &joined {
        *overall_counts.entry(taxon_id).or_default() += u64::from(count);
    }

    let mut clusters: Vec<Option<u32>> = Vec::new();
    let mut taxon_ids: Vec<u32> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut averages: Vec<f64> = Vec::new();

    for (&taxon_id, &count) in &overall_counts {
        clusters.push(None);
        taxon_ids.push(taxon_id);
        counts.push(count as u32);
        averages.push(count as f64 / total_count as f64);
    }

    // Per-cluster stats: only geocodes present in geocode_cluster_df (an
    // inner join on geocode).
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

    let mut cluster_totals: HashMap<u32, u64> = HashMap::new();
    let mut cluster_taxon_counts: HashMap<(u32, u32), u64> = HashMap::new();
    for &(geocode, taxon_id, count) in &joined {
        let Some(&cluster) = geocode_to_cluster.get(&geocode) else {
            continue;
        };
        *cluster_totals.entry(cluster).or_default() += u64::from(count);
        *cluster_taxon_counts.entry((cluster, taxon_id)).or_default() += u64::from(count);
    }

    for (&(cluster, taxon_id), &count) in &cluster_taxon_counts {
        let total_in_cluster = cluster_totals[&cluster];
        clusters.push(Some(cluster));
        taxon_ids.push(taxon_id);
        counts.push(count as u32);
        averages.push(count as f64 / total_in_cluster as f64);
    }

    let height = clusters.len();
    let cluster_col: UInt32Chunked = clusters.into_iter().collect();
    let out = DataFrame::new(
        height,
        vec![
            cluster_col.with_name("cluster".into()).into_column(),
            UInt32Chunked::from_vec("taxonId".into(), taxon_ids).into_column(),
            UInt32Chunked::from_vec("count".into(), counts).into_column(),
            Float64Chunked::from_vec("average".into(), averages).into_column(),
        ],
    )
    .map_err(to_py)?;

    Ok(PyDataFrame(out))
}
