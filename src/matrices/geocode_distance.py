from typing import Literal, TypeAlias

import numpy as np
import polars as pl
import umap
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import RobustScaler

from src.dataframes import geocode_taxa_counts
from src.logging import log_action, logger
from src.types import CompositionMetric

#: The scipy metric names this module applies to a UMAP embedding.
_EmbeddingMetric: TypeAlias = Literal["braycurtis", "euclidean"]

#: Per-metric choice of UMAP input metric and the metric used on the embedding.
#:
#: The embedding metric is not always the input metric. UMAP lays points out in
#: an ordinary Euclidean space that approximates the input metric's structure,
#: so for "presence" we hand Ward a Euclidean distance over that layout — which
#: is also what Ward linkage actually assumes. "abundance" keeps Bray-Curtis on
#: the embedding to preserve the pipeline's historical behavior, even though
#: the embedding carries negative coordinates that Bray-Curtis is not defined
#: for. That inconsistency is the reason "presence" exists.
_METRIC_CONFIG: dict[CompositionMetric, tuple[str, _EmbeddingMetric]] = {
    "abundance": ("braycurtis", "braycurtis"),
    "presence": ("dice", "euclidean"),
}


def pivot_taxon_counts(taxon_counts: pl.LazyFrame) -> pl.LazyFrame:
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
    # Get unique taxon IDs for the on_columns parameter (required for LazyFrame.pivot).
    # Sorted because `unique` does not promise an order, and an unstable column
    # order makes the feature matrix — and so any seeded UMAP fit over it —
    # irreproducible between runs.
    unique_taxon_ids = (
        taxon_counts.select("taxonId")
        .unique()
        .sort("taxonId")
        .collect(engine="streaming")
        .to_series()
    )

    return taxon_counts.pivot(
        on="taxonId",
        on_columns=unique_taxon_ids,
        index="geocode",
        values="count",
    ).sort(by="geocode")


def build_X(
    geocode_taxa_counts_lf: pl.LazyFrame,
    geocode_lf: pl.LazyFrame,
    metric: CompositionMetric = "abundance",
) -> pl.DataFrame:
    """
    Builds the feature matrix (X) for distance calculation.

    Steps:
    1. Pivot the taxon counts dataframe so rows are geocodes and columns are taxa.
    2. Fill any missing taxon counts (nulls) with 0.
    3. Assert that the order of geocodes matches the input geocode_df.
    4. Drop the 'geocode' column to keep only numerical features.
    5. Transform the counts according to `metric` — RobustScaler for
       "abundance", presence/absence binarization for "presence".

    Returns:
        A Polars DataFrame representing the transformed feature matrix.
    """
    # 1. Pivot the table
    feature_matrix = log_action(
        "Pivoting taxon counts",
        lambda: geocode_taxa_counts_lf.pipe(pivot_taxon_counts).collect(
            engine="streaming"
        ),
    )

    assert feature_matrix.height > 1, "More than one geocode is required to cluster"

    # 2. Fill nulls (taxa not present in a geocode) with 0
    feature_matrix = log_action(
        "Filling null taxon counts with 0",
        lambda: feature_matrix.fill_null(np.uint32(0)),
    )

    # 3. Ensure the order of geocodes in the matrix matches the input geocode list.
    # This is crucial for later steps that rely on matching indices.
    assert feature_matrix["geocode"].equals(
        geocode_lf.collect(engine="streaming")["geocode"]
    ), "Geocode order mismatch between pivoted matrix and geocode dataframe."

    # 4. Drop the geocode identifier column
    feature_matrix = log_action(
        "Dropping geocode column", lambda: feature_matrix.drop("geocode")
    )

    # 5. Transform counts into composition vectors
    if metric == "abundance":
        return log_action("Scaling features", lambda: feature_matrix.pipe(scale_values))
    return log_action(
        "Binarizing features to presence/absence",
        lambda: feature_matrix.pipe(binarize_values),
    )


def scale_values(feature_matrix: pl.DataFrame) -> pl.DataFrame:
    """
    Scales the feature matrix using RobustScaler.

    RobustScaler is used because it is less sensitive to outliers compared to
    StandardScaler, which is beneficial for ecological count data that might
    contain extreme values. It scales data according to the Interquartile Range (IQR).
    """
    scaler = RobustScaler()
    return pl.from_numpy(scaler.fit_transform(feature_matrix.to_numpy()))


def binarize_values(feature_matrix: pl.DataFrame) -> pl.DataFrame:
    """
    Reduces every taxon count to a 0/1 presence bit.

    Discards abundance information, which is the point: observation counts in
    citizen science data track observer effort at least as strongly as they
    track true abundance, and effort varies enormously between taxonomic
    groups. Presence is far less effort-sensitive, and unlike the scaled
    abundance path this transform is fit-free — a geocode's vector depends only
    on which taxa it contains, never on the rest of the dataset — which is what
    makes vectors from separately-run facets comparable.

    Emitted as floats rather than booleans so the result feeds UMAP and scipy
    unchanged; both read any nonzero as present.
    """
    return feature_matrix.select(pl.all().gt(0).cast(pl.Float64))


def reduce_dimensions_umap(
    X: pl.DataFrame,
    n_components: int,
    min_dist: float,
    metric: str = "braycurtis",
    random_state: int | None = None,
) -> pl.DataFrame:
    """
    Reduces the dimensionality of the feature matrix using UMAP.

    Args:
        X: The input feature matrix (Polars DataFrame).
        n_components: The number of dimensions to reduce to.
        min_dist: The minimum distance between points in the low-dimensional representation.
        metric: Dissimilarity measure UMAP uses on the input vectors.
        random_state: Seed for UMAP's layout optimization. Leaving this None
            keeps UMAP multithreaded and its output irreproducible run to run;
            setting it forces single-threaded execution in exchange for exact
            reproducibility.

    Returns:
        A Polars DataFrame with reduced dimensions.
    """
    # UMAP requires n_components to be less than the number of samples (X.height).
    # See: https://github.com/lmcinnes/umap/issues/201
    assert n_components < X.height, (
        f"UMAP requires n_components ({n_components}) to be less than "
        f"the number of samples ({X.height}). "
        f"Either reduce n_components or provide more geocodes."
    )

    reducer = umap.UMAP(
        # Target number of dimensions. Must be < number of samples.
        n_components=n_components,
        metric=metric,
        # Controls how tightly UMAP is allowed to pack points together.
        min_dist=min_dist,
        random_state=random_state,
        # Never sever an edge for being too long. UMAP defaults this to 1.0 for
        # bounded metrics including dice, and Sorensen distance is *exactly* 1
        # whenever two geocodes share no taxa — routine in sparse occurrence
        # data. Left at the default, UMAP drops those edges, fully disconnects
        # the affected geocodes, and emits NaN coordinates for them, which then
        # propagate silently through pdist and Ward into the cluster metrics.
        # Maximally dissimilar geocodes are still geocodes we want partitioned;
        # spatial contiguity is enforced by the connectivity matrix, not here.
        # This is a no-op for braycurtis, whose default is already infinite.
        disconnection_distance=np.inf,
    )
    return pl.from_numpy(reducer.fit_transform(X.to_numpy()))  # type: ignore


class GeocodeDistanceMatrix:
    """
    A distance matrix where each column and row is a geocode, and the cell at the intersection of a
    column and row is the similarity (or distance) between the two geocodes. Internally it is stored
    as a condensed distance matrix, which is a one-dimensional array containing the upper triangular
    part of the distance matrix.

    Also stores the UMAP-reduced feature matrix for computing cluster validation metrics
    like Calinski-Harabasz and Davies-Bouldin scores.
    """

    _condensed: np.ndarray
    _reduced_features: np.ndarray

    def __init__(self, condensed: np.ndarray, reduced_features: np.ndarray):
        self._condensed = condensed
        self._reduced_features = reduced_features

    @classmethod
    def build(
        cls,
        geocode_taxa_counts_lf: pl.LazyFrame,
        geocode_lf: pl.LazyFrame,
        umap_n_components: int | None = None,
        umap_min_dist: float = 0.5,
        metric: CompositionMetric = "abundance",
        random_state: int | None = None,
    ) -> "GeocodeDistanceMatrix":
        """
        Args:
            metric: How taxon counts become composition vectors. See
                `src.types.CompositionMetric`. Cross-facet comparison requires
                "presence"; "abundance" is the historical single-facet default.
            random_state: Seed for UMAP. None leaves UMAP nondeterministic.
        """
        umap_metric, embedding_metric = _METRIC_CONFIG[metric]

        # Build the feature matrix (rows=geocodes, columns=transformed taxon counts)
        feature_matrix = build_X(geocode_taxa_counts_lf, geocode_lf, metric)

        # Dimensionality Reduction using UMAP
        # UMAP is often effective for visualizing high-dimensional biological data.
        logger.info(
            f"Reducing dimensions with UMAP (metric={umap_metric}). "
            f"Input shape: {feature_matrix.shape}"
        )

        if umap_n_components is None:
            umap_n_components = feature_matrix.height - 2

        reduced_feature_matrix = log_action(
            "Fitting UMAP",
            lambda: feature_matrix.pipe(
                reduce_dimensions_umap,
                umap_n_components,
                umap_min_dist,
                umap_metric,
                random_state,
            ),
        )
        logger.info(
            f"Reduced dimensions with UMAP. Output shape: {reduced_feature_matrix.shape}"
        )

        # Calculate pairwise distances between geocodes in the reduced space.
        # pdist returns a condensed distance matrix (1D array).
        condensed_distances = log_action(
            f"Calculating pairwise distances (pdist, metric={embedding_metric}) "
            f"on matrix: {reduced_feature_matrix.shape}",
            lambda: pdist(reduced_feature_matrix, metric=embedding_metric),
        )

        return cls(condensed_distances, reduced_feature_matrix.to_numpy())

    def condensed(self) -> np.ndarray:
        return self._condensed

    def squareform(self) -> np.ndarray:
        return squareform(self._condensed)

    def reduced_features(self) -> np.ndarray:
        """
        Returns the UMAP-reduced feature matrix.

        This is needed for computing cluster validation metrics like
        Calinski-Harabasz and Davies-Bouldin scores, which require
        the feature matrix rather than the distance matrix.

        Returns:
            numpy array of shape (n_geocodes, n_components)
        """
        return self._reduced_features
