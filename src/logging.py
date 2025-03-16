import logging
import time
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

class Timer:
    """A context manager for timing code execution"""
    
    def __init__(self, output: Callable[[str], None], prefix: str = ""):
        self.output = output
        self.prefix = prefix
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        self.output(f"{self.prefix} completed in {elapsed:.4f}s")

def log_action(action: str, func: Callable[[], Any]) -> Any:
    logger.info(f"Running {action}")
    with Timer(output=logger.info, prefix=action):
        return func()
