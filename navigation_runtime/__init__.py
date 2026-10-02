"""Algorithm-neutral navigation data; no model, simulator or robot imports."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Dict, Iterator, Optional, Protocol, Sequence, Tuple, Union


@dataclass(frozen=True)
class Frame:
    """One view; depth units and the capture pose/reference are explicit."""

    image: Any = None
    path: Optional[str] = None
    depth: Any = None
    heading_degrees: float = 0.0
    color_order: str = "BGR"
    depth_unit: str = "m"
    width: Optional[int] = None
    height: Optional[int] = None
    pose: Any = None  # x/y metres and heading degrees; may retain a legacy source reference
    pose_heading_offset: Optional[float] = None  # panorama heading added when projecting the base pose
    camera_intrinsics: Any = None
    reference_frame: Optional[str] = None


@dataclass(frozen=True)
class Panorama:
    views: Dict[str, Frame]
    history: Sequence[str] = ()
    record_dir: Optional[str] = None
    origin: Optional[Frame] = None


@dataclass(frozen=True)
class MapGoal:
    """Map-cell position with explicit scale (metres/cell) and reference frame."""

    x: float
    y: float
    metres_per_cell: float
    reference_frame: str

    def __post_init__(self):
        if not all(isfinite(value) for value in (self.x, self.y, self.metres_per_cell)):
            raise ValueError("map coordinates and scale must be finite")
        if self.metres_per_cell <= 0 or not self.reference_frame:
            raise ValueError("map scale must be positive and reference frame specified")


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

    def navigate(self, goal: Union[PixelGoal, RelativeGoal], step: int,
                 on_trajectory_image: Optional[Callable[[str], None]] = None) -> ExecutionReport: ...
