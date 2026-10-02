"""Late trajectory images keep their originating task, without running hardware."""

import ast
from datetime import datetime
import math
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lavira.agent import LaViRAAgent
from lavira.runtime import Task
from navigation_runtime import ExecutionReport, Frame, Panorama


class DelayedTrajectoryHistoryTest(unittest.TestCase):
    def test_late_previous_task_callback_never_changes_next_task_or_returned_result(self):
        for profile in ("go1", "g1"):
            with self.subTest(profile=profile):
                requests, callbacks = [], []

                class Reasoning:
                    def generate_initial_todo(self, instruction, frames):
                        return "[ ] chair"

                    def language_action(self, **kwargs):
                        requests.append((kwargs["instruction"], list(kwargs["iplanner_history"])))
                        return {"action": "NAVIGATE"}

                    def vision_action(self, **kwargs):
                        return {}

                class Runtime:
                    running = True

                    def __init__(self):
                        self.history = ["older_task.jpg"]

                    def observe_panorama(self, step):
                        return Panorama({}, tuple(self.history))

                    def turn(self, degrees):
                        pass

                    def observe_current(self):
                        return Frame(image="offline")

                    def navigate(self, goal, step, on_trajectory_image=None):
                        callbacks.append(on_trajectory_image)
                        if len(callbacks) == 2:
                            # Task A's worker finishes while task B is executing.
                            self.history.append("task_A_late.jpg")
                            callbacks[0]("task_A_late.jpg")
                        path = "execution_{}.jpg".format(len(callbacks))
                        self.history.append(path)
                        on_trajectory_image(path)
                        return ExecutionReport(execution_called=True)

                runtime = Runtime()
                agent = LaViRAAgent(Reasoning(), profile)
                with patch("time.sleep"):
                    first = agent.run(Task("A", max_cycles=1), runtime)
                    second = agent.run(Task("B", max_cycles=2), runtime)
                # A callback retained by an execution worker must not mutate
                # the completed result, including after the next task ends.
                callbacks[1]("task_B_after_return.jpg")
                self.assertEqual(requests, [("A", []), ("B", []), ("B", ["execution_2.jpg"])])
                self.assertEqual(first.state.trajectory_images, ["execution_1.jpg"])
                self.assertEqual(second.state.trajectory_images, ["execution_2.jpg", "execution_3.jpg"])
                self.assertIn("task_A_late.jpg", runtime.history)

    def test_actual_go1_replan_worker_keeps_its_execution_callback(self):
        import numpy as np

        path = Path(__file__).resolve().parents[1] / "real-world-code/unitree_go1/robot/robot_controller.py"
        tree = ast.parse(path.read_text())
        controller = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef) and node.name == "RobotController")
        method = next(node for node in controller.body
                      if isinstance(node, ast.FunctionDef) and node.name == "execute_trajectory")
        pending, stops = [], []

        class DeferredThread:
            def __init__(self, target, args):
                self.target, self.args = target, args

            def start(self):
                pending.append(self)

        # Execute the real method body. Only scheduling, clocks, planning and
        # recording are replaced; no controller constructor or motion runs.
        ticks = iter((0., 1., 0., 1.))
        namespace = {
            "datetime": datetime,
            "math": math,
            "time": SimpleNamespace(time=lambda: next(ticks)),
            "threading": SimpleNamespace(Thread=DeferredThread),
            "print_action": lambda *args: None,
            "print_warning": lambda *args: None,
            "print_error": lambda *args: None,
            "print_success": lambda *args: None,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        image, depth = np.zeros((2, 2, 3)), np.ones((2, 2))
        robot = SimpleNamespace(
            running=True, planning_lock=Lock(), current_direction="front",
            capture_current_image=lambda: True,
            direction_images={"front": {"rgb": image, "depth": depth}},
            camera_data={"camera1": {"fx": None, "fy": None, "cx": None, "cy": None}},
            planner_client=SimpleNamespace(get_plan=lambda *args: ([[0., 0., 0.], [.1, 0., 0.]], .1)),
            stop_robot=lambda: stops.append("stop"),
        )
        recorded = []

        def record(*args, on_trajectory_image=None, **kwargs):
            path = "late_{}.jpg".format(len(recorded))
            recorded.append(path)
            on_trajectory_image(path)

        robot.project_trajectory_to_image = record
        first, second = [], []
        execute = namespace["execute_trajectory"]
        for images in (first, second):
            # Replanning starts before the reached-goal branch returns.
            execute(robot, [[0., 0., 0.], [.1, 0., 0.]],
                    on_trajectory_image=images.append)
        self.assertEqual(stops, ["stop", "stop"])
        self.assertEqual(len(pending), 2)
        self.assertEqual(recorded, [])
        # Newer work finishes first; the old worker still owns callback A.
        for worker in reversed(pending):
            worker.target(*worker.args)
        self.assertEqual(first, ["late_1.jpg"])
        self.assertEqual(second, ["late_0.jpg"])
