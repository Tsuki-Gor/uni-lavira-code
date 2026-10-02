"""Task-scoped algorithm data. Environment lifecycle belongs to the caller."""

from dataclasses import dataclass, field
from typing import List, Optional

from navigation_runtime import ExecutionReport


@dataclass(frozen=True)
class Task:
    instruction: str
    kind: str = "vln"
    max_cycles: Optional[int] = None


@dataclass
class TaskState:
    step: int = 1
    cycles: int = 0
    todo_list: str = ""
    visited_targets: List[dict] = field(default_factory=list)


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
