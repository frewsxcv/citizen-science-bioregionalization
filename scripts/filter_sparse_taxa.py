"""Drop very rare taxa from a cached occurrence parquet, at a fixed H3 precision.

The pipeline's own `--max-taxa` / `--min-geocode-presence` filters run *after* the
geocode set has been derived, so a filter that empties a hexagon leaves the pivoted
feature matrix with fewer geocodes than `geocode_lf` and trips the "Geocode order
mismatch" assertion in `src.matrices.geocode_distance.build_X`.

Filtering the occurrence records up front avoids that: both the geocode set and the
taxa counts are then derived from the same rows, so they agree by construction.
Hexagons whose records were all dropped disappear from the run entirely, which is the
intended outcome — a hexagon whose only observations are one-off records of taxa seen
nowhere else carries no usable composition signal.
"""

import argparse
import logging
from pathlib import Path

import polars as pl
import polars_h3

logger = logging.getLogger(__name__)


def filter_sparse_taxa_lf(
    lf: pl.LazyFrame,
    precision: int,
    min_taxon_hexes: int,
) -> pl.LazyFrame:
    """Drop occurrences of taxa observed in fewer than `min_taxon_hexes` hexagons.

    Args:
        lf: Occurrence records with `decimallatitude`, `decimallongitude` and
            `taxonkey` columns.
        precision: H3 resolution at which to measure how widespread a taxon is.
        min_taxon_hexes: Minimum number of distinct hexagons a taxon must appear
            in to be kept.

    Returns:
        The input restricted to sufficiently widespread taxa, with the same columns.
    """
    geocode = polars_h3.latlng_to_cell(
        "decimallatitude",
        "decimallongitude",
        resolution=precision,
        return_dtype=pl.UInt64,
    )
    with_geocode = lf.with_columns(geocode.alias("_geocode"))

    widespread = (
        with_geocode.group_by("taxonkey")
        .agg(pl.col("_geocode").n_unique().alias("hexes"))
        .filter(pl.col("hexes") >= min_taxon_hexes)
        .select("taxonkey")
    )

    return with_geocode.join(widespread, on="taxonkey", how="semi").drop("_geocode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Occurrence parquet to filter")
    parser.add_argument("--output", required=True, help="Where to write the result")
    parser.add_argument(
        "--precision",
        type=int,
        default=5,
        help="H3 resolution to measure taxon spread at; match the run's precision",
    )
    parser.add_argument(
        "--min-taxon-hexes",
        type=int,
        default=10,
        help="Drop taxa observed in fewer than this many distinct hexagons",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )

    source = pl.scan_parquet(args.input)
    kept = filter_sparse_taxa_lf(
        source, args.precision, args.min_taxon_hexes
    ).collect(engine="streaming")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept.write_parquet(output)

    before = source.select(pl.len()).collect().item()
    hexes = kept.select(
        polars_h3.latlng_to_cell(
            "decimallatitude",
            "decimallongitude",
            resolution=args.precision,
            return_dtype=pl.UInt64,
        ).n_unique()
    ).item()
    logger.info(
        "Wrote %d of %d rows (%.1f%%) to %s; %d taxa across %d hexagons at precision %d",
        kept.height,
        before,
        100 * kept.height / before if before else 0.0,
        output,
        kept["taxonkey"].n_unique(),
        hexes,
        args.precision,
    )


if __name__ == "__main__":
    main()
