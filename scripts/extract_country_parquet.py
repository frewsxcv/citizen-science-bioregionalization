"""Extract one country's GBIF occurrences into a local parquet cache.

The pipeline's default source is GBIF's GCS mirror, which requires Google
credentials. This script reads GBIF's AWS Open Data mirror instead, which is
anonymously readable, and narrows a snapshot down to a single country so that
repeated pipeline runs (tuning precision, k, taxonomic scope) hit a local file
rather than re-scanning the full snapshot.

Two details of the current snapshots drive the shape of this script:

1. Every file holds a single row group whose lat/lon statistics span the globe,
   so a geographic predicate prunes nothing. A bounding box can still be supplied
   to cut rows before the country comparison, but the scan is unavoidably a full
   pass over the snapshot's coordinate columns.
2. `taxonkey` is now an alphanumeric string (e.g. "3DTGL"), while the pipeline
   expects an integer (`darwin_core_utils.build_darwin_core_raw_lf` casts it to
   UInt32). Dense integer ids are assigned here and the original keys are written
   to a sidecar file, so the pipeline consumes the cache without modification.
   The ids start at SYNTHETIC_TAXON_KEY_BASE rather than 0 so that they cannot be
   mistaken for real GBIF numeric taxon keys: `dataframes.significant_taxa_images`
   looks taxa up on Wikidata by GBIF id (property P846), and small dense ids
   silently collide with unrelated real taxa, attaching a photo of a marine worm
   to an Andean bird. Out-of-range ids simply match nothing.

   This remapping is transitional. Once the pipeline accepts string taxon keys
   directly, `assign_dense_taxon_keys` and its sidecar file should be deleted and
   the snapshot's own keys carried through untouched.
"""

import argparse
import logging
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

BUCKET = "gbif-open-data-us-east-1"
LIST_ENDPOINT = f"https://{BUCKET}.s3.amazonaws.com/"
STORAGE_OPTIONS = {"aws_region": "us-east-1", "aws_skip_signature": "true"}

# Real GBIF backbone keys sit far below this, and it leaves room under UInt32's
# ceiling for far more taxa than any single country holds.
SYNTHETIC_TAXON_KEY_BASE = 3_000_000_000

# Columns the pipeline needs, in GBIF's lowercase snapshot spelling.
SOURCE_COLUMNS = [
    "decimallatitude",
    "decimallongitude",
    "scientificname",
    "taxonkey",
    "individualcount",
]


def list_snapshot_files(snapshot: str) -> list[str]:
    """List every parquet part of a snapshot via anonymous S3 listing."""
    prefix = f"occurrence/{snapshot}/occurrence.parquet/"
    keys: list[str] = []
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = LIST_ENDPOINT + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=90) as response:
            body = response.read().decode()
        keys += re.findall(r"<Key>(.*?)</Key>", body)
        match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", body)
        if match and "<IsTruncated>true</IsTruncated>" in body:
            token = match.group(1)
        else:
            break
    if not keys:
        raise ValueError(f"No parquet files found for snapshot {snapshot!r}")
    return keys


def extract_batch(
    keys: list[str],
    country: str,
    bbox: tuple[float, float, float, float],
) -> pl.DataFrame:
    """Read one batch of snapshot files down to a country's occurrences."""
    min_lat, max_lat, min_lon, max_lon = bbox
    paths = [f"s3://{BUCKET}/{key}" for key in keys]
    return (
        pl.scan_parquet(
            paths,
            storage_options=STORAGE_OPTIONS,
            parallel="prefiltered",
            low_memory=True,
        )
        .filter(
            pl.col("decimallatitude").is_not_null()
            & pl.col("decimallongitude").is_not_null()
            & pl.col("decimallatitude").is_between(min_lat, max_lat)
            & pl.col("decimallongitude").is_between(min_lon, max_lon)
            & (pl.col("countrycode") == country)
            # Records with no scientific name cannot join the taxonomy and would
            # be dropped downstream anyway.
            & pl.col("scientificname").is_not_null()
            & pl.col("taxonkey").is_not_null()
        )
        .select(SOURCE_COLUMNS)
        .collect(engine="streaming")
    )


def assign_dense_taxon_keys(lf: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.DataFrame]:
    """Replace string taxon keys with dense integers the pipeline can ingest.

    The pipeline treats taxonKey as a UInt32 identifier and joins taxa on
    (scientificName, gbifTaxonId), so any stable bijection with the snapshot's
    string keys preserves downstream behaviour. See the module docstring for why
    the ids are offset out of GBIF's real range.

    Returns:
        The frame with an integer `taxonkey`, and the mapping back to the
        snapshot's original keys.
    """
    key_map = (
        lf.select("taxonkey")
        .unique()
        .sort("taxonkey")
        .with_row_index("dense_taxonkey", offset=SYNTHETIC_TAXON_KEY_BASE)
        .collect(engine="streaming")
    )

    remapped = (
        lf.join(key_map.lazy(), on="taxonkey", how="inner")
        .drop("taxonkey")
        .rename({"dense_taxonkey": "taxonkey"})
        .with_columns(pl.col("taxonkey").cast(pl.UInt32))
        .select(SOURCE_COLUMNS)
    )
    return remapped, key_map.rename({"taxonkey": "gbif_taxonkey"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="2026-09-01", help="GBIF snapshot date")
    parser.add_argument("--country", required=True, help="ISO country code to keep")
    parser.add_argument("--output", required=True, help="Output parquet path")
    # Global by default: the bounding box is only an optional speed-up, since the
    # country code already defines the extent.
    parser.add_argument("--min-lat", type=float, default=-90.0)
    parser.add_argument("--max-lat", type=float, default=90.0)
    parser.add_argument("--min-lon", type=float, default=-180.0)
    parser.add_argument("--max-lon", type=float, default=180.0)
    parser.add_argument(
        "--batch-size", type=int, default=250, help="Snapshot files per batch"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output.parent / f".{output.stem}_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir()

    keys = list_snapshot_files(args.snapshot)
    bbox = (args.min_lat, args.max_lat, args.min_lon, args.max_lon)
    logger.info(
        "Scanning %d files from snapshot %s for country=%s",
        len(keys),
        args.snapshot,
        args.country,
    )

    started = time.time()
    total_rows = 0
    batches = [
        keys[i : i + args.batch_size] for i in range(0, len(keys), args.batch_size)
    ]
    for index, batch in enumerate(batches, start=1):
        batch_df = extract_batch(batch, args.country, bbox)
        total_rows += batch_df.height
        if batch_df.height:
            batch_df.write_parquet(shard_dir / f"shard_{index:05d}.parquet")
        elapsed = time.time() - started
        logger.info(
            "batch %d/%d  rows=%d  cumulative=%d  elapsed=%.0fs  eta=%.0fs",
            index,
            len(batches),
            batch_df.height,
            total_rows,
            elapsed,
            elapsed / index * (len(batches) - index),
        )

    if total_rows == 0:
        raise ValueError(f"No occurrences found for country {args.country!r}")

    logger.info("Assigning dense integer taxon keys over %d rows", total_rows)
    remapped, key_map = assign_dense_taxon_keys(pl.scan_parquet(shard_dir / "*.parquet"))
    logger.info("Snapshot country slice holds %d distinct taxon keys", key_map.height)

    key_map_path = output.with_name(f"{output.stem}_taxonkey_map.parquet")
    key_map.write_parquet(key_map_path)
    remapped.sink_parquet(output)
    shutil.rmtree(shard_dir)

    logger.info(
        "Wrote %d rows to %s (%.1f MiB) in %.0fs; taxon key map at %s",
        total_rows,
        output,
        output.stat().st_size / 2**20,
        time.time() - started,
        key_map_path,
    )


if __name__ == "__main__":
    main()
