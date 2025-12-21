import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)


def log_action[T](action: str, func: Callable[[], T]) -> T:
    logger.info(f"Running {action}")
    start_time = time.time()
    result = func()
    elapsed = time.time() - start_time
    logger.info(f"{action} completed in {elapsed:.4f}s")
    return result
