import numpy as np
import polars as pl
import umap
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import RobustScaler

from src.dataframes import geocode_taxa_counts
from src.logging import log_action, log_array_digest, logger

# Target dimensionality for the UMAP reduction.
#
# This used to default to `n_geocodes - 2`, which reduced almost nothing -- at
# country scale that is thousands of components -- leaving UMAP's only real
# effect the conversion of Bray-Curtis distances into Euclidean ones. It also
# silently defeated seeding: at that size the embedding goes through an
# iterative eigendecomposition that is not reproducible between processes, so
# two seeded runs on byte-identical input returned different clusterings.
# 32 is not by itself sufficient, and an earlier version of this comment
# claimed it was. What matters is n_components relative to the sample count --
# see MAX_UMAP_COMPONENT_RATIO -- because it is the spectral initialisation that
# loses reproducibility as the number of requested eigenvectors approaches the
# size of the graph. On thousands of geocodes 32 is a negligible fraction and a
# seeded run does reproduce; on 42 it is not, and does not.
DEFAULT_UMAP_N_COMPONENTS = 32

#: Largest share of the sample count that `n_components` may occupy.
#:
#: Reproducibility depends on the *ratio*, not on the absolute value. Measured
#: on the 42-geocode sample archive with a fixed seed, in one process: at 2, 4,
#: 8 and 16 components two consecutive reductions agree, and at 24 and 32 they
#: do not. 16/42 is 38% and 24/42 is 57%, so a quarter leaves margin.
#:
#: This is the same failure the old `n_samples - 2` default had, at a lower
#: threshold than "keep it small, 32" implies -- 32 is only small relative to a
#: large sample count. At country scale (thousands of geocodes) the cap binds
#: and nothing changes; on a small extent the ratio binds instead.
MAX_UMAP_COMPONENT_RATIO = 0.25


def default_umap_n_components(n_samples: int) -> int:
    """Target dimensionality for `n_samples` geocodes.

    Kept to DEFAULT_UMAP_N_COMPONENTS or a quarter of the sample count,
    whichever is smaller, and never below 2 (UMAP needs at least a plane, and
    `reduce_dimensions_umap` requires n_components < n_samples).
    """
    ratio_cap = int(n_samples * MAX_UMAP_COMPONENT_RATIO)
    return max(2, min(DEFAULT_UMAP_N_COMPONENTS, ratio_cap, n_samples - 2))


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
    # Sorted because `unique()` does not promise an order and the streaming engine
    # varies it between runs. That order becomes the feature-matrix column order,
    # and UMAP's approximate nearest-neighbour search splits on feature *indices*,
    # so an unsorted pivot makes the embedding differ run to run even with a fixed
    # seed -- which is how it defeated seeding entirely.
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
) -> pl.DataFrame:
    """
    Builds the feature matrix (X) for distance calculation.

    Steps:
    1. Pivot the taxon counts dataframe so rows are geocodes and columns are taxa.
    2. Fill any missing taxon counts (nulls) with 0.
    3. Assert that the order of geocodes matches the input geocode_df.
    4. Drop the 'geocode' column to keep only numerical features.
    5. Scale the features using RobustScaler.

    Returns:
        A Polars DataFrame representing the scaled feature matrix.
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


def reduce_dimensions_umap(
    X: pl.DataFrame,
    n_components: int,
    min_dist: float,
    random_state: int | None = None,
) -> pl.DataFrame:
    """
    Reduces the dimensionality of the feature matrix using UMAP.

    Args:
        X: The input feature matrix (Polars DataFrame).
        n_components: The number of dimensions to reduce to.
        min_dist: The minimum distance between points in the low-dimensional representation.
        random_state: Seed for UMAP's layout optimization. Leaving this None makes
            the whole pipeline nondeterministic -- the same input can yield a
            different number of clusters between runs -- so callers should pass a
            seed unless they have explicitly opted out. UMAP runs single-threaded
            once a seed is set, which is the cost of reproducibility.

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
        # Metric suitable for ecological count/abundance data.
        metric="braycurtis",
        # Controls how tightly UMAP is allowed to pack points together.
        min_dist=min_dist,
        random_state=random_state,
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
        random_state: int | None = None,
    ) -> "GeocodeDistanceMatrix":
        """
        Args:
            umap_n_components: Target dimensionality. Defaults to
                DEFAULT_UMAP_N_COMPONENTS, clamped to fit the sample count.
            random_state: Seed for UMAP. See reduce_dimensions_umap.
        """
        # Build the initial scaled feature matrix (rows=geocodes, columns=scaled taxon counts)
        scaled_feature_matrix = build_X(geocode_taxa_counts_lf, geocode_lf)

        # Dimensionality Reduction using UMAP.
        # 'braycurtis' is right *here*, where the input really is ecological
        # count data. It is not right on the output; see the pdist call below.
        logger.info(
            f"Reducing dimensions with UMAP. Input shape: {scaled_feature_matrix.shape}"
        )

        if umap_n_components is None:
            umap_n_components = default_umap_n_components(scaled_feature_matrix.height)

        # Digest either side of UMAP, so that a run which disagrees with another
        # on the final map can be localised. Matching input and differing output
        # puts the cause in UMAP; differing input puts it upstream.
        log_array_digest("umap_input", scaled_feature_matrix.to_numpy())

        reduced_feature_matrix = log_action(
            "Fitting UMAP",
            lambda: scaled_feature_matrix.pipe(
                reduce_dimensions_umap,
                umap_n_components,
                umap_min_dist,
                random_state,
            ),
        )
        logger.info(
            f"Reduced dimensions with UMAP. Output shape: {reduced_feature_matrix.shape}"
        )
        log_array_digest("umap_output", reduced_feature_matrix.to_numpy())

        # Pairwise distances between geocodes in the reduced space.
        #
        # Euclidean, not Bray-Curtis. This used to reuse Bray-Curtis "consistent
        # with the UMAP metric", which has the relationship backwards: UMAP's
        # `metric` describes the *input* space, and the embedding it returns is
        # Euclidean by construction, with signed coordinates that are not
        # abundances. On the sample archive 27.6% of embedding entries are
        # negative, and Bray-Curtis -- a ratio of summed absolute differences to
        # summed absolute totals -- has no meaning on them.
        #
        # It also matters downstream. Ward's linkage is only valid on Euclidean
        # distances, since the Lance-Williams update it uses assumes squared
        # Euclidean geometry, and the clusterer is fed this matrix directly.
        #
        # This is a correction rather than a rescue: the old metric produced no
        # NaN, no infinity and nothing outside [0, 1], and ranked pairs almost
        # identically (Spearman 0.97 against Euclidean on the same embedding).
        # Expect the map to shift rather than to be redrawn.
        condensed_distances = log_action(
            f"Calculating pairwise distances (pdist) on matrix: {reduced_feature_matrix.shape}",
            lambda: pdist(reduced_feature_matrix, metric="euclidean"),
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
