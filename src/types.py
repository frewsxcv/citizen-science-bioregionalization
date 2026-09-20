from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias, get_args

ClusterId: TypeAlias = int
Geocode: TypeAlias = str

TaxonRank: TypeAlias = Literal["kingdom", "phylum", "class", "order", "family", "genus"]

#: How a hexagon's composition is represented before distances are taken.
#:
#: "presence" records only whether a taxon was seen, "abundance" keeps the
#: counts. Presence is the default because the counts are not trustworthy:
#: `individualCount` has a median of 2 and a maximum of 35,182,100, and 19% of
#: records carry none at all and are filled with 1. A single record claiming 35
#: million individuals can dominate a hexagon's profile, which is how a moss
#: reached the top of a Colombian region's indicator taxa with 140 million.
#:
#: Bray-Curtis over presence bits is Sorensen dissimilarity, so the metric
#: itself does not change -- only what it is given.
CompositionMetric: TypeAlias = Literal["presence", "abundance", "betasim"]

#: How the composition matrix is turned into Euclidean coordinates for Ward.
#:
#: "umap" fits a manifold embedding; "pcoa" takes principal coordinates of the
#: Bray-Curtis matrix. Only the second is reproducible across machines -- see
#: matrices.geocode_distance.reduce_dimensions_pcoa.
Reduction: TypeAlias = Literal["umap", "pcoa"]

#: Agglomerative linkage rule.
#:
#: "average" is UPGMA, which Kreft & Jetz (2010) found best of nine methods and
#: which works directly on a dissimilarity matrix. "ward" needs Euclidean input,
#: which is the only reason this pipeline embeds the composition at all.
Linkage: TypeAlias = Literal["average", "ward"]

#: Ranks that can be used to scope a run, coarsest first.
TAXON_RANKS: tuple[TaxonRank, ...] = get_args(TaxonRank)

#: Darwin Core column holding the taxon name at each rank. The column is named
#: for the rank itself, so this mapping is an identity -- it exists to keep the
#: column names in one place and to document that the name column, not the
#: backbone key column, is what scoping reads.
TAXON_RANK_COLUMNS: dict[TaxonRank, str] = {
    "kingdom": "kingdom",
    "phylum": "phylum",
    "class": "class",
    "order": "order",
    "family": "family",
    "genus": "genus",
}


@dataclass(frozen=True)
class TaxonScope:
    """A taxonomic slice of the occurrence data, identified by taxon name.

    Scoping used to filter on the integer GBIF backbone key (`classKey` and
    friends), which was the better predicate: keys survive backbone releases
    that rename taxa, and integer equality prunes row groups more effectively
    than string comparison. Current GBIF snapshots no longer carry those columns
    at all -- of the backbone keys only `taxonkey` and `specieskey` remain -- so
    a key-based scope cannot run against the project's own default source. The
    rank *name* columns are still present and populated, so scoping reads those.

    The cost is homonyms. Names are only unique within a kingdom, so
    `genus:Oenanthe` matches both the wheatears and the water dropworts. A
    backbone key would not have. There is no way around this while the keys are
    absent from the data; pair the scope with a coarser rank if it matters.
    """

    rank: TaxonRank
    name: str

    @property
    def column(self) -> str:
        """The Darwin Core column this scope filters on."""
        return TAXON_RANK_COLUMNS[self.rank]

    def __str__(self) -> str:
        return f"{self.rank}:{self.name}"


@dataclass(frozen=True)
class ClusterLevels:
    """Which cuts of the merge tree a run emits, and which one it publishes.

    One object because the two numbers here used to travel separately under
    four names between them -- `default_display_level` became `published_level`
    became `display_k` became `published_k`, and `optimal_num_clusters` became
    `chosen_k` became `selector_k` -- and the distinction they encode was got
    wrong twice that way. Anything choosing a cut to build or describe wants
    `published`; `selector` is a diagnostic and decides nothing.

    Built by `hierarchy.resolve_cluster_levels`, which is the only place the
    decision is made. See `defaults.DEFAULT_DISPLAY_LEVEL` for why the two
    differ.
    """

    #: The cut every single-cut artifact is built at: the GeoJSON, the
    #: per-cluster taxa statistics, the colours, the PERMANOVA, the figures.
    published: int
    #: Where the selector's combined score peaked. Reported, never built on.
    selector: int
    #: Every cut emitted in the hierarchy, sorted. Always contains both
    #: `published` and `selector`.
    emitted: tuple[int, ...]
    #: The range the tree was actually cut at.
    min_k: int
    max_k: int
    #: Whether `selector` was asked for via `--num-clusters` rather than found.
    pinned: bool = False

    @property
    def selector_agrees(self) -> bool:
        """Whether the selector happened to land on the published cut."""
        return self.selector == self.published


class LatLng(NamedTuple):
    """A latitude/longitude coordinate pair."""

    lat: float
    lng: float


class Bbox(NamedTuple):
    """A bounding box defined by southwest and northeast corners."""

    sw: LatLng
    ne: LatLng

    @property
    def min_lat(self) -> float:
        return self.sw.lat

    @property
    def max_lat(self) -> float:
        return self.ne.lat

    @property
    def min_lng(self) -> float:
        return self.sw.lng

    @property
    def max_lng(self) -> float:
        return self.ne.lng

    @classmethod
    def from_coordinates(
        cls, min_lat: float, max_lat: float, min_lng: float, max_lng: float
    ) -> "Bbox":
        """Create a Bbox from individual coordinate values."""
        return cls(sw=LatLng(min_lat, min_lng), ne=LatLng(max_lat, max_lng))
