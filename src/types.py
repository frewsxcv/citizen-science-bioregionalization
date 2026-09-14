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
CompositionMetric: TypeAlias = Literal["presence", "abundance"]

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
