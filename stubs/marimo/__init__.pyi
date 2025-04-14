from typing import Any, TypeVar, Callable, overload, Optional

T = TypeVar('T', bound=Callable[..., Any])

# Only override the problematic class methods
class App:
    def __init__(self, width: str = ...) -> None: ...
    
    @overload
    def cell(self) -> Callable[[T], T]: ...
    
    @overload
    def cell(self, hide_code: bool = ...) -> Callable[[T], T]: ...
    
    @overload
    def cell(self, function: Optional[Callable[..., Any]] = None, *, hide_code: bool = ...) -> Any: ...
    
    def run(self) -> None: ...

# Forward all other module attributes to original module
def __getattr__(name: str) -> Any: ... 