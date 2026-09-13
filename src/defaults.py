"""Centralized default configuration values for the bioregionalization pipeline.

This module provides a single source of truth for all default parameter values.
These defaults are used by:
- Marimo UI elements (as initial values)
- CLI argument parsing (as fallbacks when args aren't provided)
"""

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
MIN_HEX_RECORDS_ABSOLUTE_FLOOR = 20
MIN_HEX_RECORDS_MEDIAN_FRACTION = 0.10
