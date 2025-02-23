from typing import Any, Self
from abc import ABC, abstractmethod


class DataContainer(ABC):
    @classmethod
    @abstractmethod
    def build(cls, *args: Any, **kwargs: Any) -> Self:
        pass
