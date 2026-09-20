"""Everything a run was asked to do, as one object.

The notebook resolved its command line and UI into roughly two dozen loose
globals and then threaded them through the cell graph by hand: a 214-line
parsing cell, a resolution cell taking 31 parameters and returning 34 names,
and every downstream cell re-declaring whichever subset it needed. Adding a
flag meant touching four cells, and a cell's signature told a reader nothing
except which settings it happened to use.

This is the same problem `types.ClusterLevels` solves for k, at a larger scale:
one concept -- what this run is -- with no single place holding it.

Frozen because a run's configuration is decided once, before any stage reads
it. A stage that wants to vary something should take it as an argument rather
than rewrite the run.
"""

from dataclasses import dataclass
from typing import Mapping, Optional

from src.types import Bbox, CompositionMetric, Linkage, Reduction, TaxonScope


@dataclass(frozen=True)
class RunConfig:
    """The resolved settings for one run of the pipeline."""

    # --- Where the data comes from ------------------------------------------
    #: Parquet file/glob, or a Darwin Core archive directory.
    parquet_source_path: str
    #: The extent to keep. Hexagons centred outside it are dropped.
    bounding_box: Bbox
    #: H3 resolution. 4-5 typical.
    geocode_precision: int
    #: Clade to restrict to, or None for everything. See src/taxon_scope.py.
    taxon_scope: Optional[TaxonScope]

    # --- How much of it to read ---------------------------------------------
    #: Cap by scan order. A biased sample, not a small one -- see README.
    limit_results: Optional[int]
    #: Approximate record count drawn uniformly. Supersedes `limit_results`.
    sample_records: Optional[int]
    #: Keep only the top N taxa by count. Off by default; it biases
    #: composition toward whichever clade is most recorded.
    max_taxa: Optional[int]
    #: Keep taxa present in at least this fraction of hexagons.
    min_geocode_presence: Optional[float]
    #: Minimum records a hexagon must hold. None derives one from the data.
    min_hex_records: Optional[int]
    #: Keep every hexagon, overriding the derived floor.
    no_hex_floor: bool
    #: Mask to the checked-in coastline, dropping sea-centred hexagons.
    terrestrial_only: bool

    # --- How the map is built -----------------------------------------------
    #: The range of k the tree is cut at.
    min_clusters_to_test: int
    max_clusters_to_test: int
    #: A k asked for via --num-clusters rather than selected.
    num_clusters_pinned: Optional[int]
    #: Combined-score weights. See defaults.METRIC_WEIGHTS.
    metric_weights: Mapping[str, float]
    #: How composition is compared, reduced and linked. These four also travel
    #: together as `types.CompositionSettings` once the run reaches clustering.
    composition_metric: CompositionMetric
    linkage: Linkage
    reduction: Reduction
    #: Seed for UMAP and PERMANOVA. None opts out of seeding.
    random_seed: Optional[int]

    # --- What comes out -----------------------------------------------------
    #: Cuts to emit, or None for the default ladder.
    hierarchy_levels: Optional[tuple[int, ...]]
    #: The cut a consumer opens on. See defaults.DEFAULT_DISPLAY_LEVEL.
    default_display_level: int
    log_file: str
    #: Skip the Wikidata lookup, the only network call after loading.
    no_images: bool
    #: Skip the findings page.
    no_findings: bool
    findings_output: str

    def as_rows(self) -> list[dict[str, object]]:
        """The settings as table rows, for the notebook's inputs table.

        Derived from the fields rather than listed by hand. The hand-written
        version had drifted: 23 rows covering 19 settings, four of them the
        bounding box's corners, and it omitted sample_records,
        default_display_level, no_images, no_findings and findings_output
        entirely -- so a reader checking what a run did could not see all of
        it, and the omissions included the setting that decides which cut gets
        published.
        """
        return [
            {
                "variable": name,
                "value": (
                    str(value)
                    if isinstance(value, (dict, Mapping))
                    else "(all taxa)"
                    if name == "taxon_scope" and value is None
                    else value
                ),
            }
            for name, value in (
                (f, getattr(self, f)) for f in self.__dataclass_fields__
            )
        ]
