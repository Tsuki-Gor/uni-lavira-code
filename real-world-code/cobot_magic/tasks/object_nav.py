"""ObjectNav uses the same navigation task, with its original default goal."""

from tasks import register_task
from .vln import VLNTask


@register_task("object_nav")
class ObjectNavTask(VLNTask):
    task_kind = "object_nav"

    def __init__(self, robot, instruction=None, max_cycles=10):
        super().__init__(robot, instruction or "find the target object", max_cycles)
