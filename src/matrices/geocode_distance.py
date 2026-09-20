import numpy as np
import polars as pl
import umap
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import RobustScaler

from src.dataframes import geocode_taxa_counts
from src.types import CompositionMetric, Reduction
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


def build_unscaled_X(
    geocode_taxa_counts_lf: pl.LazyFrame,
    geocode_lf: pl.LazyFrame,
) -> pl.DataFrame:
    """The feature matrix before scaling: raw per-geocode taxon counts.

    Pivots the counts so rows are geocodes and columns are taxa, fills the
    absences with 0, checks the row order against `geocode_lf`, and drops the
    identifier column.

    The row-order assertion is the load-bearing part: everything downstream
    addresses geocodes positionally, so a mismatch here silently relabels the
    whole map rather than failing.

    Unscaled because `build_X` centres its output with RobustScaler, which on
    data whose medians are non-zero produces negative entries -- and
    Bray-Curtis, a ratio of summed absolute differences to summed absolute
    totals, is only bounded on [0, 1] for non-negative input. On real occurrence
    data most taxa are absent from most hexagons, so the medians are zero and
    nothing goes negative; that is luck rather than a guarantee, and the
    reference metric should not depend on it.
    """
    feature_matrix = log_action(
        "Pivoting taxon counts",
        lambda: geocode_taxa_counts_lf.pipe(pivot_taxon_counts).collect(
            engine="streaming"
        ),
    )

    assert feature_matrix.height > 1, "More than one geocode is required to cluster"

    feature_matrix = feature_matrix.fill_null(np.uint32(0))

    assert feature_matrix["geocode"].equals(
        geocode_lf.collect(engine="streaming")["geocode"]
    ), "Geocode order mismatch between pivoted matrix and geocode dataframe."

    return feature_matrix.drop("geocode").cast(pl.Float64)


def build_X(
    geocode_taxa_counts_lf: pl.LazyFrame,
    geocode_lf: pl.LazyFrame,
) -> pl.DataFrame:
    """`build_unscaled_X` put through RobustScaler.

    Kept as its own entry point because the determinism suite pivots through
    it; `GeocodeDistanceMatrix.build` pivots once and scales the result
    directly, rather than calling this and pivoting a second time.
    """
    return log_action(
        "Scaling features",
        lambda: scale_values(build_unscaled_X(geocode_taxa_counts_lf, geocode_lf)),
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
    # ascontiguousarray for the same reason as in GeocodeDistanceMatrix.build:
    # Polars returns column-major and UMAP reads rows.
    return pl.from_numpy(
        reducer.fit_transform(np.ascontiguousarray(X.to_numpy()))  # type: ignore
    )


def betasim_condensed(presence: np.ndarray) -> np.ndarray:
    """Simpson's beta (betasim), the richness-independent turnover component.

        betasim = 1 - a / (min(b, c) + a)

    where `a` is the number of taxa shared by two hexagons and `b`, `c` the
    numbers unique to each. Kreft & Jetz (2010) recommend it for exactly this
    job, against Sorensen/Bray-Curtis, Jaccard and Kulczynski, on the grounds
    that those "are strongly affected by differences in species richness ... if
    there is a large difference in richness between grid cells the obtained
    values from these indices will also always be large".

    That is this pipeline's problem stated precisely. Observed richness here is
    largely sampling effort -- the two correlate at Spearman 0.975 on the
    published bounding box -- so a richness-sensitive index reports hexagons as
    dissimilar because one was visited more, not because different things live
    in them. Baselga (2010) gives the decomposition: Sorensen is turnover plus
    nestedness, and nestedness is what uneven effort manufactures, a thin
    hexagon's taxa being a subset of a well-surveyed neighbour's. betasim is
    the turnover term with nestedness removed, so a perfect subset scores 0
    rather than scoring by the size of the gap.

    Measured over 2,098 hexagons of the published bounding box:

                                  Sorensen   betasim
        median dissimilarity        0.9136    0.7072
        pairs at or above 0.95       45.0%     17.3%
        Spearman against effort      0.706     0.405

    It does not eliminate the confound -- 0.405 is still enough to align the
    top-level split with effort -- but unlike rarefying to a common depth it
    costs no hexagons, being a change of index rather than a filter.

    Args:
        presence: An (n_hexagons, n_taxa) binary matrix.

    Returns:
        Condensed pairwise dissimilarities, in `pdist` order.
    """
    shared = presence @ presence.T
    totals = presence.sum(axis=1)
    b = totals[:, None] - shared
    c = totals[None, :] - shared
    denominator = np.minimum(b, c) + shared
    with np.errstate(invalid="ignore", divide="ignore"):
        square = 1.0 - shared / denominator
    # Two hexagons sharing nothing and holding nothing leave 0/0. They are not
    # distinguishable on composition, so call them identical rather than NaN.
    square = np.nan_to_num(square, nan=0.0)
    np.fill_diagonal(square, 0.0)
    # Enforce symmetry before condensing: the division is symmetric in exact
    # arithmetic but need not be bit-for-bit, and squareform checks.
    square = (square + square.T) / 2
    return squareform(square, checks=False)


def reduce_dimensions_pcoa(
    condensed: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Principal coordinates analysis (classical MDS) of a distance matrix.

    An alternative to `reduce_dimensions_umap` with the same job -- turn an
    ecological dissimilarity into Euclidean coordinates that Ward can cluster --
    and two properties UMAP does not have.

    It is deterministic. UMAP's layout is a stochastic optimisation, and a fixed
    seed only pins it within one machine: measured on the published run, two
    processes on different machines produced byte-identical `umap_input`
    digests and different embeddings, moving the reported R2 between 0.4591 and
    0.5641 and the composition silhouette between 0.1174 and 0.1338. That
    variance is larger than most effects worth measuring here, which makes any
    single-run comparison of a pipeline change unreliable.

    It is also the ordination this field already uses. PCoA on Bray-Curtis is
    the standard treatment of a community matrix, where UMAP is a general
    manifold method whose reduction in this pipeline is close to vestigial --
    its real contribution was converting Bray-Curtis into something Euclidean,
    which is exactly what PCoA does, from an explicit eigendecomposition rather
    than a fitted layout.

    Negative eigenvalues are dropped. Bray-Curtis is not a Euclidean metric, so
    some always appear; the axes they carry have no real coordinates and are
    conventionally discarded rather than corrected.

    Args:
        condensed: Condensed pairwise dissimilarities, as returned by `pdist`.
        n_components: Maximum number of axes to keep. Fewer are returned when
            the dissimilarity supports fewer positive eigenvalues.

    Returns:
        An (n_samples, k) array of coordinates, k <= n_components.
    """
    distances = squareform(condensed)
    n = distances.shape[0]

    # Gower's double centring: B = -1/2 J D^2 J, whose eigenvectors scaled by
    # the square roots of its eigenvalues reproduce the distances.
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = centering @ (-0.5 * distances**2) @ centering
    # eigh rather than eig: the matrix is symmetric, and eigh returns real
    # values in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(gram)

    order = np.argsort(eigenvalues)[::-1][:n_components]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    keep = eigenvalues > 0
    eigenvalues, eigenvectors = eigenvalues[keep], eigenvectors[:, keep]

    # An eigenvector's sign is arbitrary, and LAPACK need not choose the same
    # one on every platform. Distances are unaffected, but the digests this
    # pipeline logs are not, so fix it: make each axis's largest-magnitude
    # entry positive.
    signs = np.sign(eigenvectors[np.argmax(np.abs(eigenvectors), axis=0), np.arange(eigenvectors.shape[1])])
    signs[signs == 0] = 1
    eigenvectors = eigenvectors * signs

    return eigenvectors * np.sqrt(eigenvalues)


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
    _abundance_condensed: np.ndarray | None

    def __init__(
        self,
        condensed: np.ndarray,
        reduced_features: np.ndarray,
        abundance_condensed: np.ndarray | None = None,
    ):
        self._condensed = condensed
        self._reduced_features = reduced_features
        self._abundance_condensed = abundance_condensed

    @classmethod
    def build(
        cls,
        geocode_taxa_counts_lf: pl.LazyFrame,
        geocode_lf: pl.LazyFrame,
        umap_n_components: int | None = None,
        umap_min_dist: float = 0.5,
        random_state: int | None = None,
        metric: CompositionMetric = "presence",
        reduction: Reduction = "umap",
    ) -> "GeocodeDistanceMatrix":
        """
        Args:
            umap_n_components: Target dimensionality. Defaults to
                DEFAULT_UMAP_N_COMPONENTS, clamped to fit the sample count.
            random_state: Seed for UMAP. See reduce_dimensions_umap.
            metric: "presence" reduces each count to whether the taxon was seen
                at all; "abundance" keeps the counts. See CompositionMetric.
            reduction: "umap" fits a manifold embedding; "pcoa" takes principal
                coordinates of the Bray-Curtis matrix. See
                reduce_dimensions_pcoa for why the second is reproducible and
                the first is not.
        """
        # Pivoted once, here, and reused by every branch below. The pivot is one
        # of the widest stages in the pipeline -- on the published extent it
        # takes its own full pass at 10-15 GB -- and it used to run twice on two
        # of the three metric paths: `presence` and `abundance` each built a
        # feature matrix and then built the reference matrix from a second,
        # identical pivot. `betasim` returns before that point, which is why the
        # default path never showed it.
        counts = build_unscaled_X(geocode_taxa_counts_lf, geocode_lf)

        if metric == "betasim":
            # Clustered on the dissimilarities themselves. Ward is what forced
            # an embedding -- it needs Euclidean input -- and UPGMA does not, so
            # with `linkage="average"` downstream there is nothing left to
            # embed and no distortion to introduce. The PCoA coordinates below
            # are computed only so Calinski-Harabasz and Davies-Bouldin have a
            # feature space to be defined in; nothing clusters on them.
            presence = (counts.to_numpy() > 0).astype(np.float64)
            logger.info(
                f"Composition metric: betasim over {presence.shape[1]} taxa "
                f"(Simpson turnover; richness-independent)"
            )
            log_array_digest("umap_input", presence)
            condensed_distances = log_action(
                f"Calculating betasim dissimilarities: {presence.shape}",
                lambda: betasim_condensed(presence),
            )
            if umap_n_components is None:
                umap_n_components = default_umap_n_components(presence.shape[0])
            reduced = log_action(
                "Taking principal coordinates (for cluster metrics only)",
                lambda: reduce_dimensions_pcoa(condensed_distances, umap_n_components),
            )
            log_array_digest("umap_output", reduced)
            # Ward clusters these, not the dissimilarities themselves, which is
            # a departure from Kreft & Jetz and is forced by something they do
            # not do: constrain clusters to be spatially contiguous. UPGMA on
            # the raw matrix is their recommendation and is available as
            # --linkage=average, but under a contiguity constraint it chains --
            # measured on 2,098 hexagons of the published bounding box it put
            # 2,052 of them in one cluster and 46 in the other at k=2, and 2,097
            # against 1 on Sorensen. A partition that peels off a fringe scores
            # well on silhouette while explaining almost no dispersion, which is
            # how it presents: silhouette 0.3491 against an R2 of 0.0031.
            #
            # Ward on the principal coordinates of the same dissimilarities
            # keeps the constraint and recovers almost all of what unconstrained
            # UPGMA finds -- at k=2, silhouette 0.3786 and R2 0.3963 against
            # UPGMA's 0.3987 and 0.4066, on balanced clusters of 1,142 and 956.
            clustering_distances = log_action(
                f"Calculating pairwise distances (pdist) on matrix: {reduced.shape}",
                lambda: pdist(reduced, metric="euclidean"),
            )
            return cls(clustering_distances, reduced, condensed_distances)

        if metric == "presence":
            # Presence deliberately skips RobustScaler. Scaling a binary column
            # is not merely pointless: for a taxon present in most hexagons the
            # median is 1, so centring maps its column to 0 and -1 -- handing
            # Bray-Curtis the negative values it is not defined for. There are
            # also no magnitudes left to normalise.
            feature_matrix = pl.from_numpy(
                (counts.to_numpy() > 0).astype(np.float64)
            )
            logger.info(
                f"Composition metric: presence/absence over {feature_matrix.width} taxa "
                f"(Bray-Curtis over presence bits is Sorensen)"
            )
        else:
            feature_matrix = log_action("Scaling features", lambda: scale_values(counts))
            logger.info(
                f"Composition metric: abundance over {feature_matrix.width} taxa"
            )
        scaled_feature_matrix = feature_matrix

        if umap_n_components is None:
            umap_n_components = default_umap_n_components(scaled_feature_matrix.height)

        # Distances on the composition itself, before any reduction. These serve
        # two purposes: they are the reference the reported silhouette is
        # checked against (see cluster_optimization, which logs both), and under
        # --reduction=pcoa they are also what the reduction is taken of, so they
        # are computed here rather than after it.
        # ascontiguousarray because Polars returns column-major and pdist walks
        # rows. Measured on Colombia, pdist over an 800-row slice took 19.0s as
        # given and 2.9s once copied -- a 6.6x penalty for nothing. End to end
        # this call went from 586s to 74s.
        reference = counts.to_numpy()
        if metric == "presence":
            reference = (reference > 0).astype(np.float64)
        unscaled = np.ascontiguousarray(reference)
        reference_space = "presence bits" if metric == "presence" else "abundances"
        abundance_condensed = log_action(
            f"Calculating reference distances (braycurtis) on "
            f"{reference_space}: {unscaled.shape}",
            lambda: pdist(unscaled, metric="braycurtis"),
        )

        # Digest either side of the reduction, so that a run which disagrees
        # with another on the final map can be localised. Matching input and
        # differing output puts the cause in the reduction; differing input puts
        # it upstream.
        log_array_digest("umap_input", scaled_feature_matrix.to_numpy())

        if reduction == "pcoa":
            logger.info(
                f"Reducing dimensions with PCoA. Input: {unscaled.shape} "
                f"({reference_space}, Bray-Curtis)"
            )
            reduced_feature_matrix = pl.from_numpy(
                log_action(
                    "Taking principal coordinates",
                    lambda: reduce_dimensions_pcoa(
                        abundance_condensed, umap_n_components
                    ),
                )
            )
        else:
            # 'braycurtis' is right *here*, where the input really is ecological
            # count data. It is not right on the output; see the pdist call
            # below.
            logger.info(
                f"Reducing dimensions with UMAP. Input shape: {scaled_feature_matrix.shape}"
            )
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
            f"Reduced dimensions with {reduction}. Output shape: {reduced_feature_matrix.shape}"
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

        return cls(
            condensed_distances,
            reduced_feature_matrix.to_numpy(),
            abundance_condensed,
        )

    def condensed(self) -> np.ndarray:
        return self._condensed

    def squareform(self) -> np.ndarray:
        return squareform(self._condensed)

    def abundance_condensed(self) -> np.ndarray | None:
        """Condensed Bray-Curtis distances on the pre-reduction abundance matrix.

        None when the matrix was constructed directly rather than via `build`,
        which is how the tests build fixtures.
        """
        return self._abundance_condensed

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
