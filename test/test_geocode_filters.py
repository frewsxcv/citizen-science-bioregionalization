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


class TestTerrestrialUsesLandShare(unittest.TestCase):
    """The mask tests the share of a hexagon's records that are on land.

    Two earlier versions tested a single synthetic point and both failed the
    same way. The hexagon's geometric centre is 17.7 km from Central Park, out
    in Long Island Sound. The marginal median of the records -- median latitude
    and median longitude taken independently -- lands in the East River, because
    Manhattan is a narrow island between two rivers. Meanwhile 88.0% of that
    cell's ten million records are on land.
    """

    MANHATTAN = (40.7812, -73.9665)   # Central Park
    EAST_RIVER = (40.7787, -73.9327)  # where the marginal median landed
    OPEN_ATLANTIC = (39.70, -71.53)

    def _records(self, points):
        return pl.DataFrame(
            {
                "decimalLatitude": [p[0] for p in points],
                "decimalLongitude": [p[1] for p in points],
            }
        ).lazy()

    def test_manhattan_is_kept(self):
        kept = filter_terrestrial_geocodes_lf(
            self._records([self.MANHATTAN] * 50), 4
        ).collect()
        self.assertEqual(kept.height, 50)

    def test_neither_synthetic_point_is_on_land(self):
        """Pins the premise. If either of these ever lands on soil, the case
        above would pass for a reason that has nothing to do with the fix."""
        tree, _ = _land_index()
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
        for label, lat, lng in (
            ("cell centre", cell["clat"][0], cell["clng"][0]),
            ("marginal median", self.EAST_RIVER[0], self.EAST_RIVER[1]),
        ):
            with self.subTest(point=label):
                hit = tree.query(shapely.points([lng], [lat]), predicate="intersects")
                self.assertEqual(len(hit[0]), 0, f"{label} is on land")

    def test_a_mostly_terrestrial_cell_survives_some_marine_records(self):
        """88% on land keeps the cell, which is Manhattan's real proportion."""
        points = [self.MANHATTAN] * 88 + [self.EAST_RIVER] * 12
        kept = filter_terrestrial_geocodes_lf(self._records(points), 4).collect()
        self.assertEqual(kept.height, 100)

    def test_a_mostly_marine_cell_is_dropped(self):
        """The mask still has to keep marine biota out, which is why it exists."""
        points = [self.MANHATTAN] * 20 + [self.EAST_RIVER] * 80
        kept = filter_terrestrial_geocodes_lf(self._records(points), 4).collect()
        self.assertEqual(kept.height, 0)

    def test_open_ocean_is_dropped(self):
        kept = filter_terrestrial_geocodes_lf(
            self._records([self.OPEN_ATLANTIC] * 50), 4
        ).collect()
        self.assertEqual(kept.height, 0)




class TestTerrestrialSamplingIsBounded(unittest.TestCase):
    """The sample is drawn per record, deterministically, without buffering.

    The previous implementation aggregated each hexagon's records and took a
    seeded shuffle of them. That is correct but holds every row of every group,
    which measured 13.9 GB at the run's old 300M-record cap -- against a 16 GB
    runner -- and is why removing the cap starved it. Sampling by a per-row
    probability instead is stateless, and holds 8.3 GB on the full 658M records.
    """

    MANHATTAN = (40.7812, -73.9665)
    EAST_RIVER = (40.7787, -73.9327)

    def _records(self, points):
        return pl.DataFrame(
            {
                "decimalLatitude": [p[0] for p in points],
                "decimalLongitude": [p[1] for p in points],
            }
        ).lazy()

    def test_sampling_is_reproducible(self):
        """Determinism is load-bearing: two runs on one input must agree
        byte-for-byte, and the sample feeds the hexagon set."""
        points = [self.MANHATTAN] * 700 + [self.EAST_RIVER] * 300
        runs = [
            filter_terrestrial_geocodes_lf(
                self._records(points), 4, sample_per_hexagon=50
            ).collect()
            for _ in range(2)
        ]
        self.assertTrue(runs[0].equals(runs[1]))

    def test_samples_records_not_distinct_locations(self):
        """Hashing coordinates rather than rows would admit a repeated point
        all-or-nothing. Here one marine location carries most of the records:
        sampled per record it is a 90% marine cell and must be dropped, while
        sampling distinct locations would see a 2-point cell that is half land.
        """
        points = [self.MANHATTAN] * 100 + [self.EAST_RIVER] * 900
        kept = filter_terrestrial_geocodes_lf(
            self._records(points), 4, sample_per_hexagon=100
        ).collect()
        self.assertEqual(kept.height, 0)

    def test_a_hexagon_smaller_than_the_sample_is_fully_used(self):
        """With n below the target the keep probability is 1, so no record is
        discarded and the share is exact rather than estimated."""
        points = [self.MANHATTAN] * 6 + [self.EAST_RIVER] * 4
        kept = filter_terrestrial_geocodes_lf(
            self._records(points), 4, sample_per_hexagon=1000
        ).collect()
        self.assertEqual(kept.height, 10)

    def test_the_sample_does_not_follow_scan_order(self):
        """The snapshot is ordered by source dataset, so a cell's leading
        records are one dataset's. Here the first 200 are marine and the
        remaining 800 terrestrial: a head-of-frame sample sees 0% land and drops
        the cell, an order-independent one sees 80% and keeps it."""
        points = [self.EAST_RIVER] * 200 + [self.MANHATTAN] * 800
        kept = filter_terrestrial_geocodes_lf(
            self._records(points), 4, sample_per_hexagon=100
        ).collect()
        self.assertEqual(kept.height, 1000)
