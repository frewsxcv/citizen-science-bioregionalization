from typing import NamedTuple
from shapely import Point


class Bbox(NamedTuple):
    sw: Point
    ne: Point
