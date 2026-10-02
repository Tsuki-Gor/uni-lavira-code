"""Algorithm-neutral navigation data; no model, simulator or robot imports."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Iterator, Optional, Protocol, Sequence, Tuple, Union


@dataclass(frozen=True)
class Frame:
    """One view. Heading is degrees to the left; depth uses metres."""

    image: Any = None
    path: Optional[str] = None
    depth: Any = None
    heading_degrees: float = 0.0
    color_order: str = "BGR"
    depth_unit: str = "m"
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass(frozen=True)
class Panorama:
    views: Dict[str, Frame]
    history: Sequence[str] = ()
    record_dir: Optional[str] = None


@dataclass(frozen=True)
class RelativeGoal:
    """Robot-local target in metres: x forward, y left."""

    x: float
    y: float

    def __post_init__(self):
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("relative coordinates must be finite metres")


@dataclass(frozen=True)
class PixelGoal:
    """Pixel in the supplied view, with an explicit optional projection fallback."""

    view: Frame
    u: float
    v: float
    fallback: Optional[RelativeGoal] = None

    def __post_init__(self):
        if not isfinite(self.u) or not isfinite(self.v):
            raise ValueError("pixel coordinates must be finite")


@dataclass(frozen=True)
class ExecutionReport:
    execution_called: bool = False
    arrived: Optional[bool] = None
    detail: str = ""


class NavigationRuntime(Protocol):
    @property
    def running(self) -> bool: ...

    def observe_panorama(self, step: int) -> Optional[Panorama]: ...

    def turn(self, degrees: float) -> None: ...

    def observe_current(self) -> Optional[Frame]: ...

    def observe_views(self) -> Iterator[Tuple[str, Frame]]: ...

    def finish_observation(self) -> None: ...

    def navigate(self, goal: Union[PixelGoal, RelativeGoal], step: int) -> ExecutionReport: ...
