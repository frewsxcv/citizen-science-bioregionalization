from typing import NamedTuple
from src.point import Point

class Bbox(NamedTuple):
    sw: Point
    ne: Point
