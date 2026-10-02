"""Object-goal navigation task.

Object Nav is essentially VLN where the instruction is "Find X"; the G1 source
defines it as a pure pass-through subclass of ``VLNTask``.
"""
from tasks import register_task
from .vln import VLNTask


@register_task("object_nav")
class ObjectNavTask(VLNTask):
    """Object Navigation Task - VLN where the instruction is 'Find X'."""

    task_kind = "object_nav"
