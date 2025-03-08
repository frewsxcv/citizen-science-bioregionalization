from typing import Any, Dict
from abc import ABC, abstractmethod
import polars as pl

class DataContainer(ABC):
    SCHEMA: Dict[str, pl.DataType]

    @classmethod
    @abstractmethod
    def build(cls, *args: Any, **kwargs: Any) -> 'DataContainer':
        pass
