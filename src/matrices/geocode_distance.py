import numpy as np
import polars as pl
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import RobustScaler

from src.dataframes import geocode_taxa_counts
from src.logging import log_action, log_array_digest, logger

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


def build_unscaled_X(
    geocode_taxa_counts_lf: pl.LazyFrame,
    geocode_lf: pl.LazyFrame,
) -> pl.DataFrame:
    """The feature matrix before scaling: raw per-geocode taxon counts.

    `build_X` centres its output with RobustScaler, which on data whose medians
    are non-zero produces negative entries -- and Bray-Curtis, a ratio of summed
    absolute differences to summed absolute totals, is only bounded on [0, 1]
    for non-negative input. On real occurrence data most taxa are absent from
    most hexagons, so the medians are zero and nothing goes negative; that is
    luck rather than a guarantee, and the reference metric should not depend on
    it.
    """
    feature_matrix = geocode_taxa_counts_lf.pipe(pivot_taxon_counts).collect(
        engine="streaming"
    )
    feature_matrix = feature_matrix.fill_null(np.uint32(0))
    assert feature_matrix["geocode"].equals(
        geocode_lf.collect(engine="streaming")["geocode"]
    ), "Geocode order mismatch between pivoted matrix and geocode dataframe."
    return feature_matrix.drop("geocode").cast(pl.Float64)


def scale_values(feature_matrix: pl.DataFrame) -> pl.DataFrame:
    """
    Scales the feature matrix using RobustScaler.

    RobustScaler is used because it is less sensitive to outliers compared to
    StandardScaler, which is beneficial for ecological count data that might
    contain extreme values. It scales data according to the Interquartile Range (IQR).
    """
    scaler = RobustScaler()
    return pl.from_numpy(scaler.fit_transform(feature_matrix.to_numpy()))


class GeocodeDistanceMatrix:
    """
    A distance matrix where each column and row is a geocode, and the cell at the intersection of a
    column and row is the similarity (or distance) between the two geocodes. Internally it is stored
    as a condensed distance matrix, which is a one-dimensional array containing the upper triangular
    part of the distance matrix.

    Also stores the feature matrix the distances were computed from, which
    Calinski-Harabasz and Davies-Bouldin need; those two are defined on a
    feature space rather than on pairwise distances.
    """

    _condensed: np.ndarray
    _features: np.ndarray

    def __init__(self, condensed: np.ndarray, features: np.ndarray):
        self._condensed = condensed
        self._features = features

    @classmethod
    def build(
        cls,
        geocode_taxa_counts_lf: pl.LazyFrame,
        geocode_lf: pl.LazyFrame,
        random_state: int | None = None,
    ) -> "GeocodeDistanceMatrix":
        """
        Args:
            random_state: Unused. Kept so callers need not know that this stage
                stopped being random when UMAP was removed; the seed still
                matters to PERMANOVA's permutations.
        """
        del random_state
        # Build the initial scaled feature matrix (rows=geocodes, columns=scaled taxon counts)
        scaled_feature_matrix = build_X(geocode_taxa_counts_lf, geocode_lf)

        # Distances straight on the counts. UMAP used to sit here, reducing to
        # 32 components before the distances were taken, and it was removed for
        # four reasons measured on real data:
        #
        #  - it explained less. Scored against these same Bray-Curtis distances,
        #    the partitions it produced had lower PERMANOVA R2 at every k above
        #    2 -- at k=4, 0.0198 against 0.0312.
        #  - it inflated the headline metric. A partition scoring silhouette
        #    0.3897 on the embedding scored 0.0564 here; across six
        #    configurations the ratio ran 3.4x to 24x, and twice the honest
        #    figure was negative while the reported one was positive.
        #  - it was the pipeline's only nondeterministic stage. Five CI runs on a
        #    byte-identical input matrix produced two different embeddings and
        #    both k=2 and k=3. Not machine-keyed, so not fixable by pinning: the
        #    digests varied between runs on one runner image.
        #  - it was not even saving time at the scale that ships. At 1155
        #    geocodes and 10000 taxa this pdist takes about 4s against UMAP's
        #    ~16s.
        #
        # UMAP is still used for the ordination plots in src/plot, where an
        # embedding is what is actually wanted.
        counts = np.ascontiguousarray(
            build_unscaled_X(geocode_taxa_counts_lf, geocode_lf).to_numpy()
        )
        logger.info(f"Computing distances on abundances. Shape: {counts.shape}")
        log_array_digest("distance_input", counts)

        # ascontiguousarray above because Polars returns column-major and pdist
        # walks rows: on Colombia that was 586s as given against 74s once copied.
        condensed_distances = log_action(
            f"Calculating pairwise distances (braycurtis) on: {counts.shape}",
            lambda: pdist(counts, metric="braycurtis"),
        )
        log_array_digest("distance_output", condensed_distances)

        return cls(condensed_distances, counts)

    def condensed(self) -> np.ndarray:
        return self._condensed

    def squareform(self) -> np.ndarray:
        return squareform(self._condensed)

    def features(self) -> np.ndarray:
        """The matrix the distances were computed from: counts per geocode.

        Calinski-Harabasz and Davies-Bouldin are defined on a feature space
        rather than on pairwise distances, and this is that space. It used to be
        a 32-column UMAP embedding; it is now the counts themselves, so the
        metrics and the distances describe the same thing.
        """
        return self._features
