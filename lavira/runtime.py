"""Task-scoped algorithm data. Environment lifecycle belongs to the caller."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from navigation_runtime import ExecutionReport


@dataclass(frozen=True)
class Task:
    instruction: str
    kind: str = "vln"
    max_cycles: Optional[int] = None
    max_steps: Optional[int] = None
    answer_from_history: bool = True
    id: Optional[str] = None
    reuse_runtime_history: bool = False  # Go1/G1 legacy entrypoints opt in.


@dataclass
class TaskState:
    step: int = 1
    cycles: int = 0
    todo_list: Union[str, List[dict], None] = ""
    visited_targets: List[dict] = field(default_factory=list)
    history_images: List[dict] = field(default_factory=list)
    total_waypoints: int = 0
    total_backtracks: int = 0
    trajectory_images: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Progress:
    stage: str
    step: int
    cycles: int


@dataclass(frozen=True)
class TaskResult:
    reason: str
    state: TaskState
    cancellation_requested: bool = False
    last_execution: Optional[ExecutionReport] = None
    physical_stop_confirmed: Optional[bool] = None
    answer: Any = None
    answer_error: Optional[str] = None
