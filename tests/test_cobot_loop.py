"""Cobot VLN/ObjectNav share a driver without changing scan/arm/motion order."""

import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from test_ground_reasoning import ROOT, original_module


class CobotLoopTest(unittest.TestCase):
    def test_vln_and_object_nav_match_original_cycles(self):
        self.assertTrue((ROOT / "navigation_runtime/gimbal_robot.py").exists(), "gimbal runtime missing")
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("requires existing image dependencies")
        Image.init()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = types.SimpleNamespace(SESSION_DIR=directory, OUTPUT_ROOT=directory)
            with patch.dict(sys.modules, {"config": types.SimpleNamespace(Config=config)}):
                helpers = original_module("cobot_magic", "utils.py")
            image = np.zeros((10, 14, 3), dtype=np.uint8)
            for task in ("vln", "object_nav"):
                for case in ("la_stop", "arrived", "not_arrived", "va_stop", "missing_bbox",
                             "missing_view", "unknown_direction", "cancel_during_la", "empty_todo"):
                    with self.subTest(task=task, case=case):
                        expected = self.run_task(True, task, case, config, helpers, image)
                        actual = self.run_task(False, task, case, config, helpers, image)
                        self.assertEqual(actual, expected)

    def run_task(self, original, task_name, case, config, helpers, image):
        trace = []
        owner = []

        class ResetThread:
            alive = True

            def start(self):
                trace.append(("arm_start",))

            def is_alive(self):
                trace.append(("arm_alive",))
                return self.alive

            def join(self):
                trace.append(("arm_join",))
                self.alive = False

        class API:
            def __init__(self, client):
                pass

            def decide_direction(self, frames, instruction, visited, todo, **kwargs):
                trace.append(("la", frames, instruction, copy.deepcopy(visited), todo, kwargs))
                if case == "cancel_during_la":
                    owner[0].stop_task()
                return {"action": "STOP" if case == "la_stop" else "NAVIGATE",
                        "updated_todo_list": "" if case == "empty_todo" else "[ ] chair",
                        "turn_direction": "unknown" if case == "unknown_direction" else "left_front",
                        "expected_landmark": "chair", "progress_analysis": "near"}

            def query_llm_bbox(self, frame, instruction, progress, visited, target, width, height, **kwargs):
                trace.append(("va", frame.tobytes(), instruction, progress, copy.deepcopy(visited),
                              target, width, height, kwargs))
                if case == "missing_bbox":
                    return None
                return {"bbox_2d": [2, 3, 10, 15], "action": "STOP" if case == "va_stop" else "NAVIGATE"}

        class Robot:
            image_lock = contextlib.nullcontext()
            rgb_width, rgb_height = 14, 10
            current_front_rgb = None if case == "missing_view" else image
            current_front_depth = None

            def collect_panoramic_images(self):
                trace.append(("panorama",))
                return [None if i == 4 else image for i in range(8)], [None] * 8, ResetThread()

            def rotate_angle(self, degrees):
                trace.append(("turn", degrees))

            def navigate_dynamic(self, u, v, **kwargs):
                trace.append(("navigate", u, v, kwargs))
                return case not in ("not_arrived", "empty_todo")

        package = types.ModuleType("tasks")
        package.__path__ = [str(ROOT / "real-world-code/cobot_magic/tasks")]
        package.register_task = lambda name: lambda cls: cls
        replacements = {
            "tasks": package, "config": types.SimpleNamespace(Config=config), "utils": helpers,
            "ai_client": types.SimpleNamespace(LaViRAVisionClient=lambda: None),
            "robot.navigation_api": types.SimpleNamespace(LaViRANavigationAPI=API),
        }
        with patch.dict(sys.modules, replacements), \
                patch("time.sleep", side_effect=lambda seconds: trace.append(("sleep", seconds))):
            if original:
                module = original_module("cobot_magic", "tasks/" + task_name + ".py")
            else:
                module = importlib.import_module("tasks." + task_name)
            cls = module.VLNTask if task_name == "vln" else module.ObjectNavTask
            task = cls(Robot(), "chair", max_cycles=2)
            owner.append(task)
            task.run()
            return trace, task.current_todo_list, task.visited_targets


if __name__ == "__main__":
    unittest.main()
