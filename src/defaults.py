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

# How taxon counts become composition vectors: "abundance" or "presence".
# See src/types.py CompositionMetric. Cross-facet comparison requires
# "presence"; single-facet runs default to the historical abundance path.
COMPOSITION_METRIC = "abundance"

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
