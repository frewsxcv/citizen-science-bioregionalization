from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias, get_args

ClusterId: TypeAlias = int
Geocode: TypeAlias = str

TaxonRank: TypeAlias = Literal["kingdom", "phylum", "class", "order", "family", "genus"]

#: Ranks that can be used to scope a run, coarsest first.
TAXON_RANKS: tuple[TaxonRank, ...] = get_args(TaxonRank)

#: Darwin Core column holding the GBIF backbone key for each rank.
TAXON_RANK_COLUMNS: dict[TaxonRank, str] = {
    "kingdom": "kingdomKey",
    "phylum": "phylumKey",
    "class": "classKey",
    "order": "orderKey",
    "family": "familyKey",
    "genus": "genusKey",
}


@dataclass(frozen=True)
class TaxonScope:
    """A taxonomic slice of the occurrence data, identified by GBIF backbone key.

    Scoping is done on the integer backbone key rather than the taxon name
    because keys are stable across backbone releases that rename taxa, and
    because an integer equality predicate pushes down into the parquet scan for
    row-group pruning far better than a string comparison does.

    `label` is carried for display and logging only; it never participates in
    filtering.
    """

    rank: TaxonRank
    key: int
    label: str

    @property
    def column(self) -> str:
        """The Darwin Core column this scope filters on."""
        return TAXON_RANK_COLUMNS[self.rank]

    def __str__(self) -> str:
        return f"{self.rank}:{self.label}({self.key})"


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
