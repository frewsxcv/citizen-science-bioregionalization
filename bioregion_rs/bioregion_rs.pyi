"""Type stub for the compiled `bioregion_rs` PyO3 extension.

Hand-maintained: PyO3/maturin extensions don't auto-generate stubs. Keep this
in sync with each `#[pyfunction]` in `src/**/*.rs` -- `pyo3_polars::PyDataFrame`
arguments/returns are plain `polars.DataFrame` at the Python boundary.
"""

import polars as pl

def darken_hex_color(hex_color: str, factor: float = 0.5) -> str: ...
def select_geocode(df: pl.DataFrame, precision: int) -> pl.DataFrame: ...
def with_geocode(df: pl.DataFrame, precision: int) -> pl.DataFrame: ...
def filter_by_bounding_box(
    df: pl.DataFrame,
    min_lat: float,
    max_lat: float,
    min_lng: float,
    max_lng: float,
    lat_col: str = "decimalLatitude",
    lng_col: str = "decimalLongitude",
) -> pl.DataFrame: ...
def build_geocode(
    df: pl.DataFrame,
    precision: int,
    min_lat: float,
    max_lat: float,
    min_lng: float,
    max_lng: float,
) -> pl.DataFrame: ...
def build_taxonomy(
    darwin_core_df: pl.DataFrame,
    geocode_precision: int,
    geocode_df: pl.DataFrame,
    min_lat: float,
    max_lat: float,
    min_lng: float,
    max_lng: float,
) -> pl.DataFrame: ...
def build_geocode_taxa_counts(
    darwin_core_df: pl.DataFrame,
    geocode_precision: int,
    taxonomy_df: pl.DataFrame,
    geocode_df: pl.DataFrame,
    min_lat: float,
    max_lat: float,
    min_lng: float,
    max_lng: float,
) -> pl.DataFrame: ...
def build_geocode_neighbors(geocode_df: pl.DataFrame) -> pl.DataFrame: ...
def build_geocode_neighbors_no_edges(
    geocode_neighbors_df: pl.DataFrame,
    geocode_no_edges_df: pl.DataFrame,
) -> pl.DataFrame: ...
def build_geocode_connectivity_matrix(
    geocode_neighbors_df: pl.DataFrame,
) -> list[list[int]]: ...
def build_cluster_taxa_statistics(
    geocode_taxa_counts_df: pl.DataFrame,
    geocode_cluster_df: pl.DataFrame,
    taxonomy_df: pl.DataFrame,
) -> pl.DataFrame: ...
def build_cluster_neighbors(
    geocode_neighbors_df: pl.DataFrame,
    geocode_cluster_df: pl.DataFrame,
) -> pl.DataFrame: ...
def build_cluster_distance_matrix(
    cluster_taxa_stats_df: pl.DataFrame,
) -> tuple[list[float], list[int]]: ...
def build_cluster_color(cluster_neighbors_df: pl.DataFrame) -> pl.DataFrame: ...
def build_cluster_boundary(
    geocode_cluster_df: pl.DataFrame,
    geocode_df: pl.DataFrame,
) -> pl.DataFrame: ...
def build_cluster_significant_differences(
    all_stats: pl.DataFrame,
    cluster_neighbors: pl.DataFrame,
) -> pl.DataFrame: ...
def build_permanova_results(
    condensed: list[float],
    geocode_ids: list[int],
    geocode_cluster_df: pl.DataFrame,
    permutations: int = 999,
) -> pl.DataFrame: ...
def build_geocode_cluster_metrics(
    condensed: list[float],
    geocode_cluster_df: pl.DataFrame,
    weight_silhouette: float = 0.4,
    weight_calinski_harabasz: float = 0.3,
    weight_davies_bouldin: float = 0.3,
) -> pl.DataFrame: ...
def select_optimal_k_elbow(
    num_clusters: list[int],
    inertia: list[float],
    sensitivity: float = 1.0,
) -> int | None: ...
def build_geocode_cluster_multi_k(
    geocodes: list[int],
    condensed: list[float],
    edge_a: list[int],
    edge_b: list[int],
    min_k: int,
    max_k: int,
) -> pl.DataFrame: ...
def build_geocode_silhouette_score(
    condensed: list[float],
    geocode_cluster_df: pl.DataFrame,
) -> pl.DataFrame: ...
def optimize_num_clusters(
    condensed: list[float],
    geocode_cluster_df: pl.DataFrame,
    elbow_sensitivity: float = 1.0,
    weight_silhouette: float = 0.4,
    weight_calinski_harabasz: float = 0.3,
    weight_davies_bouldin: float = 0.3,
) -> tuple[int, pl.DataFrame]: ...
def build_geojson_feature_collection(
    cluster_boundary_df: pl.DataFrame,
    cluster_colors_df: pl.DataFrame,
) -> str: ...
def build_json_output(
    cluster_significant_differences_df: pl.DataFrame,
    cluster_boundary_df: pl.DataFrame,
    taxonomy_df: pl.DataFrame,
    cluster_color_df: pl.DataFrame,
    significant_taxa_images_df: pl.DataFrame,
) -> str: ...
