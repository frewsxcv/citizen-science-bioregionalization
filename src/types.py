from typing import NamedTuple, TypeAlias

ClusterId: TypeAlias = int
Geocode: TypeAlias = str


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
