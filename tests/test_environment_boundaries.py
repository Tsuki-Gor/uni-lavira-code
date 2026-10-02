"""Import the concrete platform layer without creating a model or moving hardware."""

import importlib
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from threading import Lock
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentBoundaryTest(unittest.TestCase):
    def test_legacy_task_factories_run_the_shared_core(self):
        import cv2
        import numpy as np
        from PIL import Image
        from lavira.agent import LaViRAAgent
        Image.init()
        image = np.full((24, 32, 3), 100, dtype=np.uint8)
        depth = np.full((24, 32), 3000, dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for platform, profile in (("unitree_go1", "go1"), ("unitree_g1", "g1"),
                                      ("cobot_magic", "cobot")):
                with self.subTest(platform=platform), patch.dict(sys.modules):
                    for name in list(sys.modules):
                        if name.split(".")[0] in ("robot", "config", "utils", "tasks", "ai_client", "prompts"):
                            del sys.modules[name]
                    sys.modules.update({"rospy": None, "rclpy": None,
                                        "requests": types.ModuleType("requests")})
                    with patch.object(sys, "path", [str(ROOT / "real-world-code" / platform)] + sys.path):
                        config = importlib.import_module("config").Config
                        for name in ("LOG_DIR", "PANORAMA_DIR", "BBOX_DIR", "SESSION_DIR", "OUTPUT_ROOT"):
                            setattr(config, name, directory)
                        tasks = importlib.import_module("tasks")
                        for kind in ("vln", "object_nav", "eqa"):
                            effects, requests = [], []

                            class Robot:
                                running = True
                                current_direction = 0
                                current_camera = "camera1"
                                rgb_width, rgb_height = 32, 24
                                rgb_image = current_front_rgb = current_left_rgb = current_right_rgb = image
                                current_front_depth = depth
                                image_lock = Lock()
                                iplanner_history = []
                                direction_images = {0: {"rgb": image, "depth": depth}}
                                camera_data = {name: {"rgb_image": image, "fx": 32, "cx": 16}
                                               for name in ("camera1", "camera2", "camera4")}
                                planner_client = types.SimpleNamespace(get_plan=lambda *args:
                                    (np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]]), None))

                                def capture_all_directions(self, step):
                                    folder = Path(directory) / "step{}".format(step)
                                    folder.mkdir(exist_ok=True)
                                    for angle in (0, 90, 180, 270):
                                        cv2.imwrite(str(folder / "view_{}.png".format(angle)), image)
                                    return True

                                def capture_current_image(self):
                                    return True

                                def update_camera_data(self):
                                    pass

                                def get_depth_value(self, *args):
                                    return 3.

                                def project_trajectory_to_image(self, *args, **kwargs):
                                    effects.append("project")

                                def execute_trajectory(self, *args, **kwargs):
                                    effects.append("execute")

                                def collect_panoramic_images(self):
                                    reset = types.SimpleNamespace(start=lambda: None,
                                                                  is_alive=lambda: False)
                                    return [image] * 4 + [None] + [image] * 3, [depth] * 8, reset

                                def rotate_angle(self, degrees):
                                    effects.append("turn")

                                def navigate_dynamic(self, *args, **kwargs):
                                    effects.append("execute")
                                    return True

                            la = (["[ ] reach chair"] if profile == "g1" else []) + [json.dumps({
                                "action": "NAVIGATE" if profile == "cobot" else "STOP",
                                "stop": True, "turn_direction": "front", "expected_landmark": "chair",
                                "updated_todo_list": "[ ] reach chair"})]
                            if kind == "eqa":
                                la.append('{"answer":"red"}')

                            def sdk_client(role, replies):
                                def complete(**kwargs):
                                    requests.append((role, kwargs))
                                    reply = replies.pop(0)
                                    return types.SimpleNamespace(choices=[types.SimpleNamespace(
                                        message=types.SimpleNamespace(content=reply))], usage=None)
                                return types.SimpleNamespace(chat=types.SimpleNamespace(
                                    completions=types.SimpleNamespace(create=complete)))

                            clients = [sdk_client("la", la), sdk_client("va", [
                                '{"action":"NAVIGATE","bbox_2d":[100,100,900,900]}'])]
                            with self.subTest(kind=kind), patch("lavira.clients.OpenAI", side_effect=clients) as sdk, patch("time.sleep"):
                                task = tasks.TaskFactory(kind)(Robot(), "find chair")
                                self.assertEqual(sdk.call_count, 2)
                                self.assertIsInstance(task.agent, LaViRAAgent)
                                self.assertEqual(task.agent.profile, profile)
                                task.run()
                                result = task.result
                                self.assertEqual(result.reason, "todo_complete" if profile == "cobot" else "model_stop")
                                self.assertTrue(result.last_execution.execution_called)
                                self.assertEqual(effects.count("execute"), 1)
                                self.assertEqual(result.state.cycles, 1)
                                self.assertEqual(result.answer, "red" if kind == "eqa" else None)
                                self.assertIsNone(result.answer_error)
                                self.assertIsNone(result.physical_stop_confirmed)
                                expected_roles = (["la"] if profile == "g1" else []) + ["la", "va"]
                                if kind == "eqa":
                                    expected_roles.append("la")
                                self.assertEqual([role for role, _ in requests], expected_roles)
                                self.assertTrue(all(request["messages"] for _, request in requests))
                                task.stop_task()
                                self.assertTrue(task._cancellation.is_set())

    def test_robot_controller_import_does_not_load_navigation_algorithm(self):
        import cv2
        from PIL import Image
        Image.init()
        for platform in ("unitree_go1", "unitree_g1", "cobot_magic"):
            with self.subTest(platform=platform), patch.dict(sys.modules):
                for name in list(sys.modules):
                    if name.split(".")[0] in ("robot", "config", "utils", "lavira", "tasks", "ai_client"):
                        del sys.modules[name]
                # Real configuration, utility and controller modules are imported.
                # SDK/model entry points are unavailable; no controller is constructed.
                sys.modules.update({name: None for name in
                                    ("lavira", "openai", "tasks", "ai_client", "rospy", "rclpy")})
                sys.modules["requests"] = types.ModuleType("requests")
                with patch.object(sys, "path", [str(ROOT / "real-world-code" / platform)] + sys.path):
                    package = importlib.import_module("robot")
                    controller = importlib.import_module("robot.robot_controller").RobotController
                    self.assertIs(package.RobotController, controller)
                    self.assertNotIn("robot.navigation_api", sys.modules)
                    self.assertNotIn("robot.nav_controller", sys.modules)
                    with self.assertRaises(AttributeError):
                        package.missing_export
                    # Explicit requests for the old algorithm exports remain valid.
                    for module_name, export in (("navigation_api", "LaViRANavigationAPI"),
                                                ("nav_controller", "IntegratedVisionNavController")):
                        module = types.ModuleType("robot." + module_name)
                        expected = type(export, (), {})
                        setattr(module, export, expected)
                        sys.modules[module.__name__] = module
                        self.assertIs(getattr(package, export), expected)


if __name__ == "__main__":
    unittest.main()
