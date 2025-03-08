import logging
from contexttimer import Timer
from typing import Callable, Any

logger = logging.getLogger(__name__)


def log_action(action: str, func: Callable[[], Any]) -> Any:
    logger.info(f"Running {action}")
    with Timer(output=logger.info, prefix=action):
        return func()
