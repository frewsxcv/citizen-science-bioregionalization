import hashlib
import os
import tempfile

import polars as pl


def cache_darwin_core_parquet(
    darwin_core_csv_lazy_frame: pl.LazyFrame,
    input_dir: str,
) -> pl.LazyFrame:
    """
    Cache a Darwin Core LazyFrame as a parquet file for faster reloading.

    This function computes a hash of the occurrence.txt file and creates a cached
    parquet version. If the cache already exists, it loads from the cache instead
    of re-processing the CSV data.

    Args:
        darwin_core_csv_lazy_frame: The LazyFrame to cache
        input_dir: Directory containing the occurrence.txt file

    Returns:
        A new LazyFrame loaded from the cached parquet file
    """
    with open(os.path.join(input_dir, "occurrence.txt"), "rb") as f:
        file_digest = hashlib.file_digest(f, "sha256").hexdigest()

    output_dir = os.path.join(tempfile.gettempdir(), "darwin_core_cache")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_digest}.parquet")

    if not os.path.exists(output_path):
        darwin_core_csv_lazy_frame.sink_parquet(output_path)

    return pl.scan_parquet(output_path)
