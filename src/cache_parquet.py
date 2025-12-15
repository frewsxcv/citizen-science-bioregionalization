import hashlib
import os
import tempfile
from typing import TypeVar, Union, overload

import polars as pl

T = TypeVar("T", pl.LazyFrame, pl.DataFrame)


@overload
def cache_parquet(
    data: pl.LazyFrame,
    cache_key: str,
    cache_dir: str | None = None,
) -> pl.LazyFrame: ...


@overload
def cache_parquet(
    data: pl.DataFrame,
    cache_key: str,
    cache_dir: str | None = None,
) -> pl.LazyFrame: ...


def cache_parquet(
    data: Union[pl.LazyFrame, pl.DataFrame],
    cache_key: str,
    cache_dir: str | None = None,
) -> pl.LazyFrame:
    """
    Cache a Polars LazyFrame or DataFrame as a parquet file for faster reloading.

    This function creates a cached parquet version using the provided cache key.
    If the cache already exists, it loads from the cache instead of re-processing
    the data.

    Args:
        data: The LazyFrame or DataFrame to cache
        cache_key: A unique string identifier for this cache (will be hashed)
        cache_dir: Optional directory for cache files. Defaults to system temp directory.

    Returns:
        A new LazyFrame loaded from the cached parquet file
    """
    # Hash the cache key to create a consistent filename
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()

    # Set up cache directory
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "polars_cache")
    os.makedirs(cache_dir, exist_ok=True)

    output_path = os.path.join(cache_dir, f"{cache_hash}.parquet")

    # Write cache if it doesn't exist
    if not os.path.exists(output_path):
        if isinstance(data, pl.LazyFrame):
            data.sink_parquet(output_path)
        else:  # pl.DataFrame
            data.write_parquet(output_path)

    return pl.scan_parquet(output_path)
