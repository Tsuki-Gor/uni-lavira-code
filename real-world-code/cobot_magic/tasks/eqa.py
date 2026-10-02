"""EQA entrypoint using navigation and question answering from the shared core."""

from tasks import register_task
from .vln import VLNTask


@register_task("eqa")
class EQATask(VLNTask):
    task_kind = "eqa"
