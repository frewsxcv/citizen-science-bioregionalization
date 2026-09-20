"""Centralized default configuration values for the bioregionalization pipeline.

This module provides a single source of truth for all default parameter values.
These defaults are used by:
- Marimo UI elements (as initial values)
- CLI argument parsing (as fallbacks when args aren't provided)
"""

from src.types import CompositionMetric, Linkage, Reduction

# Data source defaults
PARQUET_SOURCE_PATH = (
    "gs://public-datasets-gbif/occurrence/2025-11-01/occurrence.parquet/*"
)
LOG_FILE = "run.log"

# Geocoding defaults
GEOCODE_PRECISION = 4

# Bounding box defaults (Eastern US)
MIN_LAT = 25.0
MAX_LAT = 47.0
MIN_LON = -87.0
MAX_LON = -66.0

# Clustering defaults
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20

# Seed for UMAP's layout optimization and the PERMANOVA permutation test.
# Seeded by default: unseeded runs can return a different number of clusters on
# identical input, which makes results impossible to reproduce or regression-test.
# Pass --no-seed to opt back into the faster, multithreaded UMAP path.
RANDOM_SEED: int | None = 0

# Filtering defaults
# Taxonomic scope as "rank:name" or "rank:key" (e.g. "order:Coleoptera").
# Empty means no taxonomic scoping. See src/taxon_scope.py.
TAXON_SCOPE = ""
LIMIT_RESULTS: int | None = 1000
LIMIT_RESULTS_ENABLED = True

#: Approximate number of records to keep, chosen uniformly rather than by scan
#: order. Off by default: drawing a uniform sample costs a counting pass over
#: the source, which is not what the 1000-record interactive default is for.
SAMPLE_RECORDS: int | None = None
MAX_TAXA: int | None = 5000
MAX_TAXA_ENABLED = False
MIN_GEOCODE_PRESENCE: float | None = 0.05
MIN_GEOCODE_PRESENCE_ENABLED = False
# Minimum records a hexagon must hold. None means "derive one from the data";
# see adaptive_min_hex_records. Set an integer to pin it, or pass --no-hex-floor
# to keep every hexagon.
MIN_HEX_RECORDS: int | None = None

# Parameters of the derived floor: keep hexagons holding at least a tenth of
# what the median hexagon holds, but never fewer than 20 records.
#
# Measured as mean adjusted Rand index between a partition and the same
# partition after hiding 5% of observed records, three draws per cell, k=4:
#
#                    no floor   fixed 50   max(20, 0.10*median)
#   Alps                0.975      0.926      0.984   (1094 hexagons kept)
#   California         -0.000      0.911      0.903   (1367, against 891)
#   SE Australia       -0.001      0.619      0.716   (3039, against 2579)
#   Colombia           -0.000      0.509      0.517   (3000, against 2860)
#
# Without a floor three of the four regions collapse to chance agreement: Ward
# peels under-sampled hexagons off as singleton clusters, so a partition can go
# from [1538, 1545, 790, 193] to [2460, 1604, 1, 1] on a 1% perturbation. A
# fixed 50 repairs those three but degrades the Alps, which are evenly sampled
# and need no floor. Deriving it from each region's own median does both, and
# discards fewer hexagons everywhere.
# The ceiling exists because the relative term has no natural upper bound. The
# published run is at H3 resolution 4, whose median hexagon holds roughly 73,000
# records, and a tenth of that derived a floor of 7,274 -- discarding hexagons
# with thousands of observations. Every region measured above derived between 20
# and 34, so 100 bounds that failure without changing any measured result.
# Measured against Bray-Curtis over raw counts, k=4, adaptive floor applied:
#
#                    Colombia R2   SE Australia R2   best silhouette
#   abundance             0.0351            0.0468    0.0170
#   presence              0.0591            0.1114    0.0298
#
# +68% and +138% explained variance, and the best separation of seven
# representations tried (column max, log1p, Hellinger, Wisconsin double
# standardisation and RobustScaler were the others). Presence also makes
# per-taxon scaling moot: there are no magnitudes left to normalise.
#: How hexagon composition is compared.
#:
#: betasim (Simpson turnover) rather than presence/absence Bray-Curtis, which is
#: Sorensen. Kreft & Jetz (2010) recommend it for regionalisation specifically,
#: against Sorensen/Bray-Curtis, Jaccard and Kulczynski, because those "are
#: strongly affected by differences in species richness". Here richness is
#: largely sampling effort -- the two correlate at Spearman 0.975 -- so a
#: richness-sensitive index reports hexagons as different because one was
#: visited more often. See matrices.geocode_distance.betasim_condensed.
COMPOSITION_METRIC: CompositionMetric = "betasim"

#: Cuts of the merge tree to emit, comma-separated, or None for just the one the
#: selector chose. A regionalization is conventionally reported as a nesting;
#: see src/hierarchy.py for what the levels agree with.
HIERARCHY_LEVELS: str | None = None

#: The cut a consumer opens on, which is deliberately not the one the selector
#: chose.
#:
#: The selector maximises a combined score in which silhouette carries the most
#: weight, and silhouette falls with k on saturated ecological distances, so it
#: lands near the bottom of the tested range. Not *at* it: an earlier version of
#: this note claimed the argmax is always `min_k`, which the run of 2026-09-20
#: contradicted by selecting k=3 out of a range starting at 2. The defensible
#: claim is weaker -- the selector sits low and moves under changes that have
#: nothing to do with grain. Removing the taxa cap shifted it from 2 to 3.
#:
#: Four is a presentation default, and the evidence for it is thinner than it
#: once was. On the published run, agreement with EPA Level II -- the reference
#: the pipeline is never shown -- goes:
#:
#:   k=2  ARI 0.2552    k=3  ARI 0.3107    k=4  ARI 0.3108
#:   k=8  ARI 0.2861    k=12 ARI 0.2416    k=15 ARI 0.2502
#:
#: So ARI has a real interior optimum, and it is *flat across three and four*:
#: 0.0001 apart, far below anything this measure resolves. Four is kept because
#: changing a published default on a 0.0001 difference is churn, not because it
#: won.
#:
#: The other two measures cannot arbitrate, for opposite reasons:
#:
#:   - V-measure rises almost monotonically with k here (0.3998 at k=2 to
#:     0.5098 at k=15), because it rewards subdividing the reference
#:     consistently. By V-measure the answer is always "more regions".
#:   - Clade congruence peaked at k=5 with the taxa cap and at k=3 without it,
#:     a filter with no ecological content. It is not stable enough to select
#:     on, which is why the plan to use it as the criterion was dropped.
#:
#: Read the silhouette warning beside all of this: at the published cut it is
#: 0.2382, below the 0.25 threshold, meaning the data do not show substantial
#: cluster structure at any k. None of these numbers are choosing between
#: well-separated alternatives.
#:
#: This is a presentation default, not a change to the clustering: every level
#: is still emitted and the selector's k is still computed and reported. Runs on
#: a different extent may want a different level -- `--default-level`.
DEFAULT_DISPLAY_LEVEL = 4

#: Agglomerative linkage rule. See types.Linkage.
LINKAGE: Linkage = "ward"

#: How the composition matrix becomes Euclidean coordinates for Ward.
#:
#: PCoA, because UMAP's layout is a stochastic optimisation whose seed only
#: binds within one machine. Measured on the published run, two machines
#: produced byte-identical inputs and different embeddings, moving the reported
#: R2 between 0.4591 and 0.5641 -- a spread wider than most effects this
#: pipeline is used to look for. Three PCoA runs at different seeds agree
#: exactly.
REDUCTION: Reduction = "pcoa"


# Weights for the combined score that selects k, as (silhouette,
# Calinski-Harabasz, Davies-Bouldin). Silhouette dominates because it is the
# only one of the three that expresses a preference at all.
#
# Measured on Colombia at presence/absence with the sampling floor, k from 2
# to 12: Calinski-Harabasz falls monotonically (109.39 down to 27.06) and
# Davies-Bouldin rises monotonically (4.03 up to 5.97). A monotone criterion
# carries no information about the right k -- it votes for the end of the
# range, which is why k=2 was chosen every time despite silhouette peaking at
# k=6 and turning negative by k=11.
#
# With these weights and the normalisation fixed, Colombia selects k=6. Any
# silhouette weight at or above 0.6 does; below that the two monotone metrics
# win again.
METRIC_WEIGHTS: dict[str, float] = {
    "silhouette": 0.7,
    "calinski_harabasz": 0.15,
    "davies_bouldin": 0.15,
}

MIN_HEX_RECORDS_ABSOLUTE_FLOOR = 20
MIN_HEX_RECORDS_MEDIAN_FRACTION = 0.10
MIN_HEX_RECORDS_CEILING = 100

#: Where the findings page is written. Inside the gitignored output directory,
#: because it is an artifact of a run rather than a document -- every number on
#: it is recomputed, so a committed copy would go stale silently.
FINDINGS_OUTPUT_PATH = "output/findings.html"
