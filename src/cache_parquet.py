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
    cache_key: str,
    cache_dir: str | None = None,
) -> dy.LazyFrame[SchemaT]: ...


@overload
def cache_parquet(
    data: dy.DataFrame[SchemaT],
    cache_key: str,
    cache_dir: str | None = None,
) -> dy.LazyFrame[SchemaT]: ...


def cache_parquet(
    data: dy.LazyFrame[SchemaT] | dy.DataFrame[SchemaT],
    cache_key: str,
    cache_dir: str | None = None,
) -> dy.LazyFrame[SchemaT]:
    # Hash the cache key to create a consistent filename
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()

    # Set up cache directory
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "polars_cache")
    os.makedirs(cache_dir, exist_ok=True)

    output_path = os.path.join(cache_dir, f"{cache_hash}.parquet")

    if isinstance(data, pl.LazyFrame):
        data.sink_parquet(output_path)
    else:
        data.write_parquet(output_path)

    return cast(dy.LazyFrame[SchemaT], pl.scan_parquet(output_path))
