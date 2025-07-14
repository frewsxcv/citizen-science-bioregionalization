import os

import numpy as np
import polars as pl
from sklearn.decomposition import IncrementalPCA  # type: ignore
from sklearn.preprocessing import RobustScaler  # type: ignore
from scipy.spatial.distance import pdist, squareform  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
import umap  # type: ignore
from src.data_container import DataContainer
from src.dataframes import geocode_taxa_counts
from src.dataframes.geocode import GeocodeDataFrame
from src.logging import log_action, logger


def pivot_taxon_counts(taxon_counts: pl.DataFrame) -> pl.DataFrame:
    """
    Create a matrix where each row is a geocode and each column is a taxon ID

    Example input:

    ```txt
    ┌─────────┬─────────┬───────┐
    │ geocode ┆ taxonId ┆ count │
    │ ---     ┆ ---     ┆ ---   │
    │ str     ┆ u32     ┆ u32   │
    ╞═════════╪═════════╪═══════╡
    │ 9eu     ┆ 12345   ┆ 1     │
    │ 9ev     ┆ 23456   ┆ 1     │
    │ 9ev     ┆ 34567   ┆ 1     │
    │ 9ev     ┆ 45678   ┆ 2     │
    │ 9ev     ┆ 56789   ┆ 4     │
    └─────────┴─────────┴───────┘
    ```

    Example output:

    ```txt
    ┌─────────┬───────┬───────┬───────┬───┬───────┐
    │ geocode ┆ 12345 ┆ 23456 ┆ 34567 ┆ … ┆ 56789 │
    │ ---     ┆ ---   ┆ ---   ┆ ---   ┆   ┆ ---   │
    │ str     ┆ u32   ┆ u32   ┆ u32   ┆   ┆ u32   │
    ╞═════════╪═══════╪═══════╪═══════╪═══╪═══════╡
    │ 9eu     ┆ 1     ┆ null  ┆ null  ┆ … ┆ null  │
    │ 9ev     ┆ null  ┆ 1     ┆ 1     ┆ … ┆ 4     │
    └─────────┴───────┴───────┴───────┴───┴───────┘
    ```
    """
    return taxon_counts.pivot(
        on="taxonId",
        index="geocode",
        values="count",
    )


def build_X(
    geocode_taxa_counts_dataframe: geocode_taxa_counts.GeocodeTaxaCountsDataFrame,
    geocode_dataframe: GeocodeDataFrame,
) -> pl.DataFrame:
    """
    Builds the feature matrix (X) for distance calculation.

    Steps:
    1. Pivot the taxon counts dataframe so rows are geocodes and columns are taxa.
    2. Fill any missing taxon counts (nulls) with 0.
    3. Assert that the order of geocodes matches the input geocode_dataframe.
    4. Drop the 'geocode' column to keep only numerical features.
    5. Scale the features using RobustScaler.

    Returns:
        A Polars DataFrame representing the scaled feature matrix.
    """
    # 1. Pivot the table
    feature_matrix = log_action(
        "Pivoting taxon counts",
        lambda: geocode_taxa_counts_dataframe.df.pipe(pivot_taxon_counts),
    )

    assert feature_matrix.height > 1, "More than one geocode is required to cluster"

    # 2. Fill nulls (taxa not present in a geocode) with 0
    feature_matrix = log_action(
        "Filling null taxon counts with 0",
        lambda: feature_matrix.fill_null(np.uint32(0)),
    )

    # 3. Ensure the order of geocodes in the matrix matches the input geocode list.
    # This is crucial for later steps that rely on matching indices.
    assert (
        feature_matrix["geocode"].to_list() == geocode_dataframe.df["geocode"].to_list()
    ), "Geocode order mismatch between pivoted matrix and geocode dataframe."

    # 4. Drop the geocode identifier column
    feature_matrix = log_action(
        "Dropping geocode column", lambda: feature_matrix.drop("geocode")
    )

    # 5. Scale features
    scaled_feature_matrix = log_action(
        "Scaling features", lambda: feature_matrix.pipe(scale_values)
    )

    return scaled_feature_matrix


def scale_values(feature_matrix: pl.DataFrame) -> pl.DataFrame:
    """
    Scales the feature matrix using RobustScaler.

    RobustScaler is used because it is less sensitive to outliers compared to
    StandardScaler, which is beneficial for ecological count data that might
    contain extreme values. It scales data according to the Interquartile Range (IQR).
    """
    scaler = RobustScaler()
    return pl.from_numpy(scaler.fit_transform(feature_matrix.to_numpy()))


def reduce_dimensions_umap(X: pl.DataFrame) -> pl.DataFrame:
    """
    Reduces the dimensionality of the feature matrix using UMAP.

    Args:
        X: The input feature matrix (Polars DataFrame).

    Returns:
        A Polars DataFrame with reduced dimensions.
    """
    # UMAP requires n_components to be less than the number of samples (X.height).
    # Setting it to X.height - 2 provides some margin.
    # See: https://github.com/lmcinnes/umap/issues/201
    # The assertion ensures there are enough samples for the chosen n_components.
    # TODO: Revisit this constraint for smaller datasets if necessary.
    assert (
        X.height > 10
    ), "UMAP requires more samples for the current n_components setting."

    reducer = umap.UMAP(
        # Target number of dimensions. Must be < number of samples.
        n_components=X.height - 2,
        # Metric suitable for ecological count/abundance data.
        metric="braycurtis",
        # Controls how tightly UMAP is allowed to pack points together.
        min_dist=0.5,
    )
    return pl.from_numpy(reducer.fit_transform(X.to_numpy()))


class GeocodeDistanceMatrix(DataContainer):
    """
    A distance matrix where each column and row is a geocode, and the cell at the intersection of a
    column and row is the similarity (or distance) between the two geocodes. Internally it is stored
    as a condensed distance matrix, which is a one-dimensional array containing the upper triangular
    part of the distance matrix.
    """

    _condensed: np.ndarray

    def __init__(self, condensed: np.ndarray):
        self._condensed = condensed

    @classmethod
    def build(
        cls,
        geocode_taxa_counts_dataframe: geocode_taxa_counts.GeocodeTaxaCountsDataFrame,
        geocode_dataframe: GeocodeDataFrame,
    ) -> "GeocodeDistanceMatrix":
        # Build the initial scaled feature matrix (rows=geocodes, columns=scaled taxon counts)
        scaled_feature_matrix = build_X(
            geocode_taxa_counts_dataframe, geocode_dataframe
        )

        # Dimensionality Reduction using UMAP
        # UMAP is often effective for visualizing high-dimensional biological data.
        # 'braycurtis' is chosen as the metric because it's suitable for ecological count data (abundance data).
        logger.info(
            f"Reducing dimensions with UMAP. Input shape: {scaled_feature_matrix.shape}"
        )
        reduced_feature_matrix = log_action(
            "Fitting UMAP",
            lambda: scaled_feature_matrix.pipe(reduce_dimensions_umap),
        )
        logger.info(
            f"Reduced dimensions with UMAP. Output shape: {reduced_feature_matrix.shape}"
        )

        # Calculate pairwise distances between geocodes in the reduced space
        # Using 'braycurtis' distance again, consistent with the UMAP metric.
        # pdist returns a condensed distance matrix (1D array).
        condensed_distances = log_action(
            f"Calculating pairwise distances (pdist) on matrix: {reduced_feature_matrix.shape}",
            lambda: pdist(reduced_feature_matrix, metric="braycurtis"),
        )

        return cls(condensed_distances)

    def condensed(self) -> np.ndarray:
        return self._condensed

    def squareform(self) -> np.ndarray:
        return squareform(self._condensed)
