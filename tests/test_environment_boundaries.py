"""Import the concrete platform layer without creating a model or moving hardware."""

import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentBoundaryTest(unittest.TestCase):
    def test_legacy_task_factories_assemble_the_shared_core(self):
        import cv2
        from PIL import Image
        from lavira.agent import LaViRAAgent
        from lavira.runtime import TaskResult, TaskState
        Image.init()
        with tempfile.TemporaryDirectory() as directory:
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
                            with self.subTest(kind=kind), patch("lavira.clients.OpenAI") as sdk:
                                task = tasks.TaskFactory(kind)(object(), "find chair")
                                self.assertEqual(sdk.call_count, 2)
                                self.assertIsInstance(task.agent, LaViRAAgent)
                                self.assertEqual(task.agent.profile, profile)
                                result = TaskResult("model_stop", TaskState(), answer="chair")
                                with patch.object(LaViRAAgent, "run", return_value=result) as run:
                                    task.run()
                                request, runtime = run.call_args.args
                                self.assertEqual((request.kind, request.instruction), (kind, "find chair"))
                                self.assertIs(runtime.robot, task.robot)
                                self.assertIs(task.result, result)
                                task.stop_task()
                                self.assertTrue(run.call_args.kwargs["cancellation"].is_set())

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
