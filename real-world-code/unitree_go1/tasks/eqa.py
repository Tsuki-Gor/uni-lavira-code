"""EQA entrypoint using navigation and question answering from the shared core."""

from tasks import register_task
from .vln import VLNTask
from config import Config
from utils import print_error, save_output


@register_task("eqa")
class EQATask(VLNTask):
    task_kind = "eqa"

    def run(self):
        super().run()
        if self.result.answer_error is None:
            try:
                save_output(Config.LOG_DIR, "final_eqa_result.json", {
                    "question": self.instruction, "answer": self.result.answer,
                })
            except Exception as error:
                print_error("EQA Reasoning failed: {}".format(error))
