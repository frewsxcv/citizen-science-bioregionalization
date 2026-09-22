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
    label: str,
    spill_dir: str | None = None,
) -> pl.LazyFrame:
    """Spill a frame to parquet and hand back a lazy scan of it.

    This is not a cache. It always writes, and never reads back a previous
    run's file. It exists to cut a long lazy query graph into stages so that
    memory is released between them.

    The parameters used to be called `cache_key` and `cache_dir`, which said
    the opposite of what the docstring underneath them had to keep correcting.

    Args:
        data: The frame to write.
        label: Human-readable name for the stage, used in the filename and in
            logs. Need only be unique within a run.
        spill_dir: Directory to write into. Defaults to a per-run subdirectory
            of DATA_DIR (or the system temp directory).

    Returns:
        A LazyFrame scanning the file just written.
    """
    # Hash the label to keep the filename filesystem-safe and fixed-length.
    name_hash = hashlib.sha256(label.encode()).hexdigest()

    if spill_dir is None:
        # Use DATA_DIR environment variable if set (persistent disk on GCP),
        # otherwise fall back to system temp directory
        base_dir = os.environ.get("DATA_DIR", tempfile.gettempdir())
        spill_dir = os.path.join(base_dir, "polars_intermediates", _RUN_ID)
        _run_dirs.add(spill_dir)
    os.makedirs(spill_dir, exist_ok=True)

    output_path = os.path.join(spill_dir, f"{name_hash}.parquet")

    if isinstance(data, pl.LazyFrame):
        logger.info(f"Spilling {label} LazyFrame to {output_path}")
        data.sink_parquet(output_path, engine="streaming")
    else:
        logger.info(f"Spilling {label} DataFrame to {output_path}")
        data.write_parquet(output_path)

    result = pl.scan_parquet(output_path)

    # Read from the parquet footer rather than by scanning, so this is cheap.
    # Worth logging at every stage: until now nothing reported how many records
    # a run actually ingested, which left --limit-results unfalsifiable -- there
    # was no way to tell a cap that bound from one that never came near binding.
    row_count = result.select(pl.len()).collect().item()
    logger.info(f"Materialized {label}: {row_count} rows at {output_path}")

    return result
