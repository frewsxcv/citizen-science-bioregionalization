import csv
from typing import Dict, Generator, NamedTuple, Optional, Self
import logging

import pygeohash
from src.point import Point

logger = logging.getLogger(__name__)

type TaxonId = int


class Row(NamedTuple):
    location: Point
    taxon_id: TaxonId
    scientific_name: str
    # TODO: Should this be a user ID?
    observer: str

    @classmethod
    def from_csv_dict(cls, row: Dict[str, str]) -> Optional[Self]:
        lat, lon = read_float(row["decimalLatitude"]), read_float(
            row["decimalLongitude"]
        )
        if not (lat and lon):
            logger.error(
                f"Invalid latitude or longitude: {row['decimalLatitude']}, {row['decimalLongitude']}"
            )
            return None
        taxon_id = read_int(row["taxonKey"])
        if not taxon_id:
            logger.error(f"Invalid taxon ID: {row['taxonKey']}")
            return None
        return cls(Point(lat, lon), taxon_id, row["scientificName"], row["recordedBy"])

    def geohash(self, precision: int) -> str:
        return pygeohash.encode(
            self.location.lat, self.location.lon, precision=precision
        )


def read_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def read_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None

def read_rows(input_file: str) -> Generator[Row, None, None]:
    with open(input_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for dict_row in reader:
            row = Row.from_csv_dict(dict_row)
            if row:
                yield row