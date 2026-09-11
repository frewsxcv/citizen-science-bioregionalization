import hashlib
import logging
import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def log_action[T](action: str, func: Callable[[], T]) -> T:
    logger.info(f"Running {action}")
    start_time = time.time()
    result = func()
    elapsed = time.time() - start_time
    logger.info(f"{action} completed in {elapsed:.4f}s")
    return result


def log_array_digest(label: str, array: np.ndarray) -> None:
    """Log a sha256 of an array's contents, for localising nondeterminism.

    Two runs that agree on every row count can still disagree on the map, and
    when that happens the counts in the log cannot say which stage diverged.
    A digest at each numeric stage can: matching input with differing output
    localises the cause to the step between them.

    This exists because two CI runs with provably identical inputs -- same
    257253279 rows ingested, same 1598228 rows and 10000 taxa after filtering,
    same 1155 geocodes into UMAP, seed applied in both -- chose k=2 and k=3.
    The runs used different runner images, and GitHub-hosted runners cannot be
    pinned to an exact image, so the question has to be answered from inside
    the pipeline.

    Digests the raw bytes, so it is sensitive to dtype and shape as well as to
    values, and cheap even for the feature matrix (tens of MB, well under a
    second).
    """
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    logger.info(
        f"digest {label}: sha256={digest[:16]} shape={array.shape} dtype={array.dtype}"
    )
