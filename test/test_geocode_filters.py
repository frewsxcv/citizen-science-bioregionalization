"""Tests for the two occurrence-level geocode filters.

Both deliberately filter occurrence records rather than the geocode set, so that
the geocodes and the taxa counts are derived from the same rows.
"""

import unittest

import polars as pl
import shapely
import polars_h3

from src.geocode import _land_index, adaptive_min_hex_records, filter_sparse_geocodes_lf, filter_terrestrial_geocodes_lf

PRECISION = 5


def _occurrences(points: list[tuple[float, float]]) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "decimalLatitude": [p[0] for p in points],
            "decimalLongitude": [p[1] for p in points],
            "scientificName": [f"Genus sp{i}" for i in range(len(points))],
        }
    )


class TestFilterSparseGeocodes(unittest.TestCase):
    def test_drops_hexagons_below_the_record_floor(self):
        # Three records in one hexagon, one in another far away.
        lf = _occurrences([(4.0, -74.0), (4.0, -74.0), (4.0, -74.0), (-2.0, -70.0)])

        result = filter_sparse_geocodes_lf(lf, PRECISION, min_records=2).collect()

        self.assertEqual(result.height, 3)

    def test_floor_of_one_keeps_everything(self):
        points = [(4.0, -74.0), (-2.0, -70.0)]
        lf = _occurrences(points)

        result = filter_sparse_geocodes_lf(lf, PRECISION, min_records=1).collect()

        self.assertEqual(result.height, len(points))

    def test_preserves_columns(self):
        lf = _occurrences([(4.0, -74.0), (4.0, -74.0)])

        result = filter_sparse_geocodes_lf(lf, PRECISION, min_records=1)

        self.assertEqual(result.collect_schema().names(), lf.collect_schema().names())


class TestFilterTerrestrialGeocodes(unittest.TestCase):
    def test_keeps_inland_records(self):
        # Bogota, the Llanos, and the Amazon.
        lf = _occurrences([(4.7, -74.1), (4.5, -71.0), (-1.0, -70.0)])

        result = filter_terrestrial_geocodes_lf(lf, PRECISION).collect()

        self.assertEqual(result.height, 3)

    def test_drops_open_ocean_records(self):
        # Open Caribbean and Pacific, well away from any coast.
        lf = _occurrences([(13.0, -76.0), (4.0, -79.5)])

        result = filter_terrestrial_geocodes_lf(lf, PRECISION).collect()

        self.assertEqual(result.height, 0)

    def test_separates_land_from_sea_in_one_frame(self):
        lf = _occurrences([(4.7, -74.1), (13.0, -76.0), (-1.0, -70.0)])

        result = filter_terrestrial_geocodes_lf(lf, PRECISION).collect()

        self.assertEqual(
            sorted(result["decimalLatitude"].to_list()), [-1.0, 4.7]
        )

    def test_preserves_columns(self):
        lf = _occurrences([(4.7, -74.1)])

        result = filter_terrestrial_geocodes_lf(lf, PRECISION)

        self.assertEqual(result.collect_schema().names(), lf.collect_schema().names())


if __name__ == "__main__":
    unittest.main()


class TestAdaptiveMinHexRecords(unittest.TestCase):
    """The floor is derived per region because no fixed value serves all.

    Measured as mean adjusted Rand index between a partition and the same
    partition after hiding 5% of records, three draws, k=4: without a floor the
    Alps hold at 0.975 while California, southeast Australia and Colombia all
    sit at chance. A flat 50 repairs those three and drops the Alps to 0.926.
    max(20, 0.10 * median) scores 0.984 / 0.903 / 0.716 / 0.517 and keeps more
    hexagons than the flat floor in every one of them.
    """

    def _records(self, per_hexagon: list[int]) -> pl.LazyFrame:
        """One frame with the requested record count in each of several hexagons."""
        rows = []
        for i, n in enumerate(per_hexagon):
            # Spread hexagons far enough apart to land in distinct H3 cells.
            lat, lng = 10.0 + i * 0.5, -70.0 + i * 0.5
            rows.extend({"decimalLatitude": lat, "decimalLongitude": lng} for _ in range(n))
        return pl.DataFrame(rows).lazy()

    def test_scales_with_the_median_when_sampling_is_dense(self):
        floor = adaptive_min_hex_records(
            self._records([1000, 800, 1200, 900]), 5, absolute_floor=20,
            median_fraction=0.10, ceiling=100,
        )
        self.assertEqual(floor, 95)  # median 950, a tenth of it

    def test_absolute_floor_wins_when_the_whole_region_is_thin(self):
        """California's median hexagon held 25 records, so a purely relative
        floor came out at 2 and filtered nothing."""
        floor = adaptive_min_hex_records(
            self._records([25, 30, 20, 25]), 5, absolute_floor=20,
            median_fraction=0.10, ceiling=100,
        )
        self.assertEqual(floor, 20)

    def test_a_long_sparse_tail_does_not_drag_the_floor_down(self):
        """The median is used rather than the mean precisely so that a tail of
        one-record hexagons -- the thing being filtered -- cannot set the
        threshold that filters it."""
        dense = [500] * 10
        tail = [1] * 9
        floor = adaptive_min_hex_records(
            self._records(dense + tail), 5, absolute_floor=20, median_fraction=0.10, ceiling=100,
        )
        self.assertEqual(floor, 50)

    def test_returns_the_absolute_floor_for_an_empty_frame(self):
        empty = pl.DataFrame(
            {"decimalLatitude": [], "decimalLongitude": []},
            schema={"decimalLatitude": pl.Float64, "decimalLongitude": pl.Float64},
        ).lazy()
        self.assertEqual(
            adaptive_min_hex_records(
                empty, 5, absolute_floor=20, median_fraction=0.10, ceiling=100
            ),
            20,
        )

    def test_ceiling_caps_a_densely_surveyed_region(self):
        """Found in CI, not in testing. The published run is at H3 resolution 4,
        whose median hexagon holds around 73,000 records; a tenth of that derived
        a floor of 7,274, which discards hexagons holding thousands of
        observations. All four regions used to choose the rule were resolution 5
        and derived 20 to 34, so none of them exposed it."""
        floor = adaptive_min_hex_records(
            self._records([73000, 70000, 75000]), 4, absolute_floor=20,
            median_fraction=0.10, ceiling=100,
        )
        self.assertEqual(floor, 100)


class TestTerrestrialUsesRecordPositions(unittest.TestCase):
    """The mask represents a hexagon by its records, not its geometric centre.

    The centre was the original test. At H3 resolution 4 a cell spans roughly
    1,770 km2, so its midpoint can sit 18 km from the records -- and Manhattan's
    cell centres at 40.8584, -73.7819, out in Long Island Sound. One of the most
    intensively recorded hexagons on the published map was being discarded, and
    107 land-containing cells with it, 8.2% of them, biased toward the coast
    where recording is densest.
    """

    MANHATTAN = (40.7812, -73.9665)   # Central Park
    OPEN_ATLANTIC = (39.70, -71.53)   # south of Long Island, genuinely at sea

    def _records(self, points: list[tuple[float, float]]) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "decimalLatitude": [p[0] for p in points],
                "decimalLongitude": [p[1] for p in points],
            }
        ).lazy()

    def test_manhattan_survives_though_its_cell_centre_is_at_sea(self):
        kept = filter_terrestrial_geocodes_lf(
            self._records([self.MANHATTAN] * 20), 4
        ).collect()
        self.assertEqual(kept.height, 20, "Manhattan's hexagon was dropped again")

    def test_the_cell_centre_really_is_at_sea(self):
        """Pins the premise. If this ever fails the case above proves nothing,
        because the cell would be passing for the trivial reason."""
        cell = (
            pl.DataFrame({"lat": [self.MANHATTAN[0]], "lng": [self.MANHATTAN[1]]})
            .with_columns(
                polars_h3.latlng_to_cell(
                    "lat", "lng", resolution=4, return_dtype=pl.UInt64
                ).alias("c")
            )
            .with_columns(
                clat=polars_h3.cell_to_lat("c"), clng=polars_h3.cell_to_lng("c")
            )
        )
        tree, _ = _land_index()
        centre = shapely.points([cell["clng"][0]], [cell["clat"][0]])
        self.assertEqual(
            len(tree.query(centre, predicate="intersects")[0]),
            0,
            "the cell centre is on land, so this no longer tests anything",
        )

    def test_open_ocean_is_still_dropped(self):
        """The mask exists to keep marine biota out; it must still do that."""
        kept = filter_terrestrial_geocodes_lf(
            self._records([self.OPEN_ATLANTIC] * 20), 4
        ).collect()
        self.assertEqual(kept.height, 0)

    def test_a_cell_is_judged_by_where_most_of_its_records_are(self):
        """A median, not a mean, so a cell holding a dense city plus some
        offshore records resolves to the city rather than to a midpoint that is
        in neither. The offshore points are Manhattan's own cell centre, which
        is in Long Island Sound, so they are certain to share its hexagon."""
        offshore = (40.8584, -73.7819)
        mostly_city = [self.MANHATTAN] * 18 + [offshore] * 2
        records = self._records(mostly_city)
        cells = (
            records.with_columns(
                polars_h3.latlng_to_cell(
                    "decimalLatitude", "decimalLongitude",
                    resolution=4, return_dtype=pl.UInt64,
                ).alias("c")
            )
            .select("c")
            .unique()
            .collect()
        )
        self.assertEqual(cells.height, 1, "the fixture spans more than one hexagon")

        kept = filter_terrestrial_geocodes_lf(records, 4).collect()
        self.assertEqual(kept.height, 20)
