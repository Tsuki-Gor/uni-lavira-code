"""Run original and shared navigation with recorded model/robot effects."""

import contextlib
import copy
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
import types
import unittest
from unittest.mock import patch

from test_ground_reasoning import ROOT, original_module


class NavigationLoopTest(unittest.TestCase):
    def test_eqa_and_object_nav_use_the_shared_complete_task(self):
        from lavira.runtime import TaskResult, TaskState
        self.assertTrue(hasattr(TaskResult("model_stop", TaskState()), "answer"),
                        "EQA answer missing from complete-task result")
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("requires existing image dependencies")
        Image.init()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = types.SimpleNamespace(PANORAMA_DIR=directory, SAFE_DISTANCE=1.2, LOG_DIR=directory)
            image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
            step_dir = Path(directory) / "step1"
            step_dir.mkdir()
            for angle in (0, 90, 180, 270):
                cv2.imwrite(str(step_dir / ("view_" + str(angle) + ".png")), image)
            for platform in ("unitree_go1", "unitree_g1"):
                helpers = original_module(platform, "utils.py")
                for task_kind in ("object_nav", "eqa"):
                    cases = ("stop_no_bbox", "cancel_during_la")
                    if task_kind == "eqa":
                        cases += ("qa_failure", "qa_missing_views", "qa_raw")
                    for case in cases:
                        with self.subTest(platform=platform, task=task_kind, case=case):
                            expected = self.run_task(platform, True, case, config, helpers, image, np, task_kind)
                            actual = self.run_task(platform, False, case, config, helpers, image, np, task_kind)
                            self.assertEqual(actual, expected)

    def test_complete_task_state_progress_and_cancel_are_separate_from_arrival(self):
        try:
            from lavira.agent import LaViRAAgent
        except ImportError:
            self.skipTest("requires existing image dependencies")
        from lavira.runtime import Task
        from navigation_runtime import Frame, Panorama, PixelGoal, RelativeGoal

        class API:
            def language_action(self, **kwargs):
                self.todo = kwargs["todo_list"]
                return {"action": "STOP", "updated_todo_list": "[x] chair"}

            def vision_action(self, **kwargs):
                return {}

        class Runtime:
            running = True
            observations = 0

            def observe_panorama(self, step):
                self.observations += 1
                return Panorama({})

            def turn(self, degrees):
                self.angle = degrees

            def observe_current(self):
                return Frame(image="offline-frame")

        agent = LaViRAAgent(API(), "go1")
        runtime = Runtime()
        progress = []
        with patch("time.sleep"):
            first = agent.run(Task("chair"), runtime, on_progress=progress.append)
            second = agent.run(Task("another chair"), runtime)
            event = Event()
            event.set()
            cancelled = agent.run(Task("chair"), runtime, cancellation=event)
            limited = agent.run(Task("chair", max_cycles=0), runtime)
        self.assertEqual(first.reason, "model_stop")
        self.assertIsNone(first.physical_stop_confirmed)
        self.assertIsNone(first.last_execution)
        self.assertIsNot(first.state, second.state)
        self.assertEqual(agent.reasoning.todo, "No TODO list yet. Please generate one.")
        self.assertEqual([p.stage for p in progress], ["started", "observing", "reasoning", "finished"])
        self.assertEqual(runtime.observations, 2)
        self.assertEqual(cancelled.reason, "cancelled")
        self.assertTrue(cancelled.cancellation_requested)
        self.assertIsNone(cancelled.physical_stop_confirmed)
        self.assertEqual(limited.reason, "budget_exhausted")
        agent.reasoning.client = types.SimpleNamespace(
            generate_with_la=lambda *args, **kwargs: ('{"answer":"red"}', {}))
        runtime.observe_views = lambda: iter(())
        progress.clear()
        with patch("time.sleep"), contextlib.redirect_stdout(io.StringIO()):
            answered = agent.run(Task("What color?", kind="eqa"), runtime,
                                 cancellation=event, on_progress=progress.append)
        self.assertEqual(answered.reason, "cancelled")
        self.assertTrue(answered.cancellation_requested)
        self.assertEqual(answered.answer, "red")
        self.assertIsNone(answered.answer_error)
        self.assertEqual([p.stage for p in progress], ["started", "answering", "finished"])
        for make_goal in (lambda: RelativeGoal(float("nan"), 0),
                          lambda: PixelGoal(Frame(), 0, float("inf"))):
            with self.assertRaises(ValueError):
                make_goal()

        # A different algorithm may import and use execution without LaViRA.
        with patch.dict(sys.modules, {"lavira": None, "lavira.agent": None}):
            module = importlib.import_module("navigation_runtime.camera_robot")
            module = importlib.reload(module)
            turns = []
            robot = types.SimpleNamespace(rotate_left=lambda a: turns.append(a),
                                          rotate_right=lambda a: turns.append(-a))
            adapter = module.CameraRobotRuntime(robot, "", (), 1.5)
            adapter.turn(45)
            adapter.turn(-90)
            with self.assertRaises(ValueError):
                adapter.turn(float("inf"))
            self.assertEqual(turns, [45, -90])

    def test_camera_platforms_keep_action_order_and_final_approach(self):
        self.assertTrue((ROOT / "lavira/agent.py").exists(), "shared task loop missing")
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("requires existing image dependencies")
        Image.init()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = types.SimpleNamespace(PANORAMA_DIR=directory, SAFE_DISTANCE=1.2)
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            for step in (1, 2):
                step_dir = Path(directory) / ("step" + str(step))
                step_dir.mkdir()
                for angle in (0, 90, 180, 270):
                    cv2.imwrite(str(step_dir / ("view_" + str(angle) + ".png")), image)
            for platform in ("unitree_go1", "unitree_g1"):
                helpers = original_module(platform, "utils.py")
                for case in ("navigate", "strategic_stop", "va_stop", "stop_no_bbox",
                             "missing_bbox", "failed_plan", "cancel_during_la", "failed_capture"):
                    with self.subTest(platform=platform, case=case):
                        expected = self.run_task(platform, True, case, config, helpers, image, np)
                        actual = self.run_task(platform, False, case, config, helpers, image, np)
                        self.assertEqual(actual, expected)

    def run_task(self, platform, original, case, config, helpers, image, np, task_kind="vln"):
        trace = []
        owner = []
        if task_kind == "eqa":
            answer_file = Path(config.LOG_DIR) / "final_eqa_result.json"
            if answer_file.exists():
                answer_file.unlink()

        class API:
            def __init__(self, client):
                self.client = client

            def generate_initial_todo(self, instruction, frames):
                trace.append(("todo", instruction, frames))
                return "[ ] chair"

            def language_action(self, **kwargs):
                trace.append(("la", copy.deepcopy(kwargs)))
                if case == "cancel_during_la":
                    owner[0].stop_task()
                return {"action": "STOP" if case in ("strategic_stop", "stop_no_bbox", "failed_plan") else "NAVIGATE",
                        "stop": "true" if case in ("strategic_stop", "stop_no_bbox", "failed_plan") else False,
                        "turn_direction": "left", "expected_landmark": "chair",
                        "reasoning": "visible", "progress_analysis": "near", "updated_todo_list": "[x] chair"}

            def vision_action(self, **kwargs):
                # The frame identity is recorded separately; arrays cannot be
                # compared with normal tuple equality.
                frame = kwargs.pop("img_np")
                trace.append(("va", kwargs, frame.shape, frame.tobytes()))
                result = {"action": "STOP" if case == "va_stop" else "NAVIGATE"}
                if case not in ("stop_no_bbox", "missing_bbox"):
                    result["bbox_2d"] = [100, 200, 400, 700]
                return result

        class Robot:
            running = True
            current_direction = 0
            current_camera = "front"
            iplanner_history = ["old.jpg"]

            def __init__(self):
                self.direction_images = {0: {"rgb": image, "depth": np.ones((480, 640))}}
                self.camera_data = {"front": {"fx": 500, "cx": 320}}
                self.rgb_image = None if case == "qa_missing_views" else image
                for index, name in enumerate(("camera1", "camera2", "camera4")):
                    self.camera_data[name] = {"rgb_image": None if case == "qa_missing_views"
                                             else np.roll(image, index, axis=0)}
                self.planner_client = types.SimpleNamespace(get_plan=self.get_plan)

            def update_camera_data(self):
                trace.append(("refresh_qa",))

            def capture_all_directions(self, step):
                trace.append(("panorama", step))
                return True

            def capture_current_image(self):
                trace.append(("current",))
                if case == "failed_capture":
                    self.running = False
                    return False
                return True

            def rotate_left(self, angle):
                trace.append(("left", angle))

            def rotate_right(self, angle):
                trace.append(("right", angle))

            def get_depth_value(self, x, y, direction):
                trace.append(("depth", x, y, direction))
                return 2.0

            def get_plan(self, rgb, depth, goal):
                trace.append(("plan", goal))
                self.running = False
                return (None if case == "failed_plan" else [[0, 0], [1, 0], [2, 0], [3, 0]]), 0.0

            def project_trajectory_to_image(self, traj, rgb, **kwargs):
                trace.append(("visualize", kwargs))

            def execute_trajectory(self, trajectory, goal_local):
                trace.append(("execute", trajectory, goal_local))

        class Client:
            def generate_with_la(self, messages, **kwargs):
                trace.append(("qa", copy.deepcopy(messages), kwargs))
                if case == "qa_failure":
                    raise RuntimeError("offline QA failure")
                return ("raw answer" if case == "qa_raw" else '{"answer":"red"}'), {}

        package = types.ModuleType("tasks")
        package.__path__ = [str(ROOT / "real-world-code" / platform / "tasks")]
        package.register_task = lambda name: lambda cls: cls
        replacements = {
            "config": types.SimpleNamespace(Config=config),
            "utils": helpers,
            "ai_client": types.SimpleNamespace(LaViRAVisionClient=Client),
            "robot.navigation_api": types.SimpleNamespace(LaViRANavigationAPI=API),
            "tasks": package,
            "colorama": types.SimpleNamespace(Fore=helpers.Fore, init=lambda **kw: None),
        }
        with patch.dict(sys.modules, replacements), \
                patch("time.sleep", side_effect=lambda seconds: trace.append(("sleep", seconds))), \
                patch("time.strftime", return_value="123456"):
            if original:
                module = original_module(platform, "tasks/vln.py")
            else:
                spec = importlib.util.spec_from_file_location(
                    platform + "_task", ROOT / "real-world-code" / platform / "tasks/vln.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            cls = module.VLNTask
            if task_kind != "vln":
                sys.modules["tasks.vln"] = module
                if original:
                    entry = original_module(platform, "tasks/" + task_kind + ".py", package="tasks")
                else:
                    spec = importlib.util.spec_from_file_location(
                        "tasks." + task_kind, ROOT / "real-world-code" / platform / "tasks" / (task_kind + ".py"))
                    entry = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(entry)
                cls = entry.EQATask if task_kind == "eqa" else entry.ObjectNavTask
            task = cls(Robot(), "find the chair")
            owner.append(task)
            task.run()
            if task_kind == "eqa":
                saved = json.loads(answer_file.read_text()) if answer_file.exists() else None
                self.assertEqual(saved, None if case == "qa_failure" else {
                    "question": "find the chair", "answer": "raw answer" if case == "qa_raw" else "red"})
                trace.append(("answer_file", saved))
            if not original:
                self.assertIsNone(task.result.physical_stop_confirmed)
                if task.result.last_execution is not None:
                    self.assertIsNone(task.result.last_execution.arrived)
                self.assertEqual(task.result.cancellation_requested, case == "cancel_during_la")
                if task_kind == "eqa":
                    self.assertEqual(task.result.answer, None if case == "qa_failure" else
                                     "raw answer" if case == "qa_raw" else "red")
                    self.assertEqual(task.result.answer_error, "offline QA failure" if case == "qa_failure" else None)
            return trace, task.current_step, task.todo_list


if __name__ == "__main__":
    unittest.main()
