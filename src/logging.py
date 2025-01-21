import logging
from contexttimer import Timer
from typing import Callable

logger = logging.getLogger(__name__)


def log_action[T](action: str, func: Callable[[], T]) -> T:
    logger.info(f"Running {action}")
    with Timer(output=logger.info, prefix=action):
        return func()
