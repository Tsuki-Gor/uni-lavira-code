"""Task memory is separate from the runtime's accumulated execution records."""

import importlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lavira.agent import LaViRAAgent
from lavira.runtime import Task
from navigation_runtime import ExecutionReport, Frame, Panorama


class TaskHistoryTest(unittest.TestCase):
    def test_camera_execution_records_only_its_images_without_lavira(self):
        import numpy as np
        from navigation_runtime import PixelGoal

        image = np.zeros((6, 8, 3), dtype=np.uint8)
        depth = np.full((6, 8), 2000, dtype=np.uint16)
        history = ["earlier_task.jpg"]
        effects = []
        records = []

        def execute(trajectory, **kwargs):
            effects.append("execute")
            history.append("replan.jpg")
            kwargs["on_trajectory_image"]("replan.jpg")

        def project(*args, **kwargs):
            history.append("plan.jpg")
            kwargs["on_trajectory_image"]("plan.jpg")

        robot = SimpleNamespace(
            iplanner_history=history, current_direction=0, current_camera="front",
            capture_current_image=lambda: True,
            direction_images={0: {"rgb": image, "depth": depth}},
            get_depth_value=lambda *args: 2., camera_data={"front": {"fx": 8, "cx": 4}},
            planner_client=SimpleNamespace(get_plan=lambda *args: ([[0, 0], [1, 0], [2, 0]], None)),
            project_trajectory_to_image=project,
            execute_trajectory=execute,
        )
        with patch.dict(sys.modules, {"lavira": None, "lavira.agent": None}):
            module = importlib.reload(importlib.import_module("navigation_runtime.camera_robot"))
            runtime = module.CameraRobotRuntime(robot, "", (), 0.5)
            goal = PixelGoal(runtime.observe_current(), 4, 5)
            report = runtime.navigate(goal, 1, on_trajectory_image=records.append)
            self.assertTrue(report.execution_called)
            self.assertEqual(records, ["plan.jpg", "replan.jpg"])
            self.assertEqual(effects, ["execute"])
            runtime.safety_distance = 3.
            records.clear()
            buffered = runtime.navigate(goal, 2, on_trajectory_image=records.append)
            self.assertFalse(buffered.execution_called)
            self.assertEqual(records, ["plan.jpg"])
            robot.planner_client.get_plan = lambda *args: (None, None)
            records.clear()
            failed = runtime.navigate(goal, 3, on_trajectory_image=records.append)
            self.assertEqual(records, [])
            self.assertEqual(effects, ["execute"])
            self.assertEqual(history[0], "earlier_task.jpg")

    def test_new_tasks_isolate_history_and_legacy_tasks_keep_it(self):
        for profile in ("go1", "g1"):
            for legacy in (False, True):
                with self.subTest(profile=profile, legacy=legacy):
                    requests = []

                    class Reasoning:
                        def generate_initial_todo(self, instruction, frames):
                            return "[ ] chair"

                        def language_action(self, **kwargs):
                            requests.append(list(kwargs["iplanner_history"]))
                            stop = kwargs["current_step"] == 2
                            return {"action": "STOP" if stop else "NAVIGATE", "stop": stop}

                        def vision_action(self, **kwargs):
                            return {}

                    class Runtime:
                        running = True

                        def __init__(self):
                            self.history = ["earlier_task.jpg"]

                        def observe_panorama(self, step):
                            return Panorama({}, tuple(self.history))

                        def turn(self, degrees):
                            pass

                        def observe_current(self):
                            return Frame(image="offline")

                        def navigate(self, goal, step, on_trajectory_image=None):
                            path = "execution_{}.jpg".format(len(self.history))
                            self.history.append(path)
                            if on_trajectory_image is not None:
                                on_trajectory_image(path)
                            return ExecutionReport(execution_called=True)

                    runtime = Runtime()
                    agent = LaViRAAgent(Reasoning(), profile)
                    options = {"reuse_runtime_history": True} if legacy else {}
                    with patch("time.sleep"):
                        first = agent.run(Task("chair", **options), runtime)
                        second = agent.run(Task("another chair", **options), runtime)
                        limited = agent.run(Task("one execution", max_cycles=1, **options), runtime)
                    if legacy:
                        self.assertEqual(requests, [
                            ["earlier_task.jpg"], ["earlier_task.jpg", "execution_1.jpg"],
                            ["earlier_task.jpg", "execution_1.jpg"],
                            ["earlier_task.jpg", "execution_1.jpg", "execution_2.jpg"],
                            ["earlier_task.jpg", "execution_1.jpg", "execution_2.jpg"],
                        ])
                    else:
                        self.assertEqual(requests, [[], ["execution_1.jpg"], [], ["execution_2.jpg"], []])
                        self.assertEqual(first.state.trajectory_images, ["execution_1.jpg"])
                        self.assertEqual(second.state.trajectory_images, ["execution_2.jpg"])
                        self.assertEqual(limited.state.trajectory_images[-1], "execution_3.jpg")
                    self.assertEqual(limited.reason, "budget_exhausted")
                    self.assertEqual(runtime.history, ["earlier_task.jpg", "execution_1.jpg",
                                                       "execution_2.jpg", "execution_3.jpg"])
                    self.assertIsNot(first.state.trajectory_images, second.state.trajectory_images)
