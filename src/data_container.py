from typing import Any
from abc import ABC, abstractmethod


class DataContainer(ABC):
    @classmethod
    @abstractmethod
    def build(cls, *args: Any, **kwargs: Any) -> 'DataContainer':
        pass
