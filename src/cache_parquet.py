import hashlib
import logging
import os
import tempfile

import polars as pl

logger = logging.getLogger(__name__)


def cache_parquet(
    data: pl.LazyFrame | pl.DataFrame,
    cache_key: str,
    cache_dir: str | None = None,
) -> pl.LazyFrame:
    # Hash the cache key to create a consistent filename
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()

    # Set up cache directory
    if cache_dir is None:
        # Use DATA_DIR environment variable if set (persistent disk on GCP),
        # otherwise fall back to system temp directory
        base_dir = os.environ.get("DATA_DIR", tempfile.gettempdir())
        cache_dir = os.path.join(base_dir, "polars_cache")
    os.makedirs(cache_dir, exist_ok=True)

    output_path = os.path.join(cache_dir, f"{cache_hash}.parquet")

    if isinstance(data, pl.LazyFrame):
        logger.info(f"Writing data from {cache_key} LazyFrame to {output_path}")
        data.sink_parquet(output_path, engine="streaming")
    else:
        logger.info(f"Writing data from {cache_key} DataFrame to {output_path}")
        data.write_parquet(output_path)

    return pl.scan_parquet(output_path)
