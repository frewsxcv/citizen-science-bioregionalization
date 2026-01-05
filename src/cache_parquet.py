import hashlib
import os
import tempfile
from typing import TypeVar, cast, overload

import dataframely as dy
import polars as pl

SchemaT = TypeVar("SchemaT", bound=dy.Schema)


@overload
def cache_parquet(
    data: dy.LazyFrame[SchemaT],
    cache_key: type[dy.Schema],
    cache_dir: str | None = None,
) -> dy.LazyFrame[SchemaT]: ...


@overload
def cache_parquet(
    data: dy.DataFrame[SchemaT],
    cache_key: type[dy.Schema],
    cache_dir: str | None = None,
) -> dy.LazyFrame[SchemaT]: ...


def cache_parquet(
    data: dy.LazyFrame[SchemaT] | dy.DataFrame[SchemaT],
    cache_key: type[dy.Schema],
    cache_dir: str | None = None,
) -> dy.LazyFrame[SchemaT]:
    # Hash the cache key to create a consistent filename
    cache_hash = hashlib.sha256(cache_key.__name__.encode()).hexdigest()

    # Set up cache directory
    if cache_dir is None:
        # Use DATA_DIR environment variable if set (persistent disk on GCP),
        # otherwise fall back to system temp directory
        base_dir = os.environ.get("DATA_DIR", tempfile.gettempdir())
        cache_dir = os.path.join(base_dir, "polars_cache")
    os.makedirs(cache_dir, exist_ok=True)

    output_path = os.path.join(cache_dir, f"{cache_hash}.parquet")

    if isinstance(data, pl.LazyFrame):
        data.sink_parquet(output_path, engine="streaming")
    else:
        data.write_parquet(output_path)

    return cast(dy.LazyFrame[SchemaT], pl.scan_parquet(output_path))
