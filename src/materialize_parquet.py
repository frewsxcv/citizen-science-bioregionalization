import atexit
import hashlib
import logging
import os
import shutil
import tempfile
import uuid

import polars as pl

logger = logging.getLogger(__name__)

# Intermediates are namespaced per process. The previous scheme derived the
# filename from the schema name alone, so two pipeline runs sharing a machine
# (or a mounted DATA_DIR) wrote to the same paths and silently overwrote each
# other's intermediates mid-run.
_RUN_ID = uuid.uuid4().hex[:12]

_run_dirs: set[str] = set()


def _cleanup_run_dirs() -> None:
    """Remove this process's intermediates on exit.

    Without this the per-run namespacing would leak a directory per run, which
    matters most on the persistent DATA_DIR disk where nothing else prunes them.
    Anything that still needs the data has already collected it by interpreter
    shutdown.
    """
    for path in _run_dirs:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_run_dirs)


def materialize_parquet(
    data: pl.LazyFrame | pl.DataFrame,
    cache_key: str,
    cache_dir: str | None = None,
) -> pl.LazyFrame:
    """Spill a frame to parquet and hand back a lazy scan of it.

    Despite the historical name this is not a cache: it always writes, and never
    reads back a previous run's file. It exists to cut a long lazy query graph
    into stages so that memory is released between them.

    Args:
        data: The frame to write.
        cache_key: Human-readable label for the frame, used in the filename and
            in logs. Need only be unique within a run.
        cache_dir: Directory to write into. Defaults to a per-run subdirectory
            of DATA_DIR (or the system temp directory).

    Returns:
        A LazyFrame scanning the file just written.
    """
    # Hash the key to keep the filename filesystem-safe and fixed-length.
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()

    if cache_dir is None:
        # Use DATA_DIR environment variable if set (persistent disk on GCP),
        # otherwise fall back to system temp directory
        base_dir = os.environ.get("DATA_DIR", tempfile.gettempdir())
        cache_dir = os.path.join(base_dir, "polars_intermediates", _RUN_ID)
        _run_dirs.add(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    output_path = os.path.join(cache_dir, f"{cache_hash}.parquet")

    if isinstance(data, pl.LazyFrame):
        logger.info(f"Writing data from {cache_key} LazyFrame to {output_path}")
        data.sink_parquet(output_path, engine="streaming")
    else:
        logger.info(f"Writing data from {cache_key} DataFrame to {output_path}")
        data.write_parquet(output_path)

    return pl.scan_parquet(output_path)
