"""Compare prompt/image payloads and parsing against each original profile."""

import contextlib
import copy
import importlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def original_module(platform, filename, package=None):
    source = subprocess.check_output([
        "git", "show", "215e7ac:real-world-code/{}/{}".format(platform, filename),
    ], cwd=ROOT, text=True)
    module = types.ModuleType("original_" + filename.replace("/", "_").replace(".", "_"))
    if package is not None:
        module.__package__ = package
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


class RecordingClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate_with_la(self, messages, **kwargs):
        self.calls.append(("la", copy.deepcopy(messages), kwargs))
        return self.reply, {"output_speed": 1, "prompt_speed": 2, "duration": 3}

    def generate_with_va(self, messages, **kwargs):
        self.calls.append(("va", copy.deepcopy(messages), kwargs))
        return self.reply, {"output_speed": 1, "prompt_speed": 2, "duration": 3}


class GroundReasoningRegressionTest(unittest.TestCase):
    def test_profile_payloads_and_bbox_results_match_release(self):
        self.assertTrue((ROOT / "lavira/reasoning").is_dir(), "independent reasoning missing")
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("requires existing image dependencies: NumPy and OpenCV")
        # Load Pillow's decoder registry before temporarily replacing project
        # modules; restoring sys.modules must not remove lazily loaded plugins.
        Image.init()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = types.SimpleNamespace(VISUAL_HISTORY_SIZE=2, LOG_DIR=directory, BBOX_DIR=directory)
            image = np.arange(1200 * 1600 * 3, dtype=np.uint8).reshape(1200, 1600, 3)
            history = [str(Path(directory) / name) for name in
                       ("replan_old.jpg", "plain.jpg", "replan_new.jpg")]
            for path in history:
                cv2.imwrite(path, image)
            for platform, profile in (("unitree_go1", "go1"), ("unitree_g1", "g1"),
                                      ("cobot_magic", "cobot")):
                with self.subTest(profile=profile):
                    with patch.dict(sys.modules, {
                        "config": types.SimpleNamespace(Config=config),
                        "ai_client": types.SimpleNamespace(LaViRAVisionClient=RecordingClient),
                    }):
                        prompts = original_module(platform, "prompts.py")
                        helpers = original_module(platform, "utils.py")
                        spec = importlib.util.spec_from_file_location(
                            profile + "_utils", ROOT / "real-world-code" / platform / "utils.py")
                        compatibility_helpers = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(compatibility_helpers)
                        with patch.dict(sys.modules, {"prompts": prompts, "utils": helpers}):
                            baseline = original_module(platform, "robot/navigation_api.py")
                    with patch.dict(sys.modules, {"config": None, "robot": None, "habitat": None,
                                                 "utils": None, "prompts": None, "ai_client": None}):
                        shared = importlib.import_module("lavira.reasoning." + profile)

                    # Real encoders catch resizing, colour order and file format drift.
                    names = (("image_to_base64",) if profile == "cobot" else
                             ("numpy_to_base64", "img_to_base64"))
                    for name in names:
                        value = history[0] if name == "img_to_base64" else image
                        encoded = getattr(shared, name)(value)
                        self.assertTrue(encoded)
                        self.assertEqual(encoded, getattr(helpers, name)(value))
                        self.assertEqual(encoded, getattr(compatibility_helpers, name)(value))
                    for reply in (
                        '{"action":"NAVIGATE","turn_direction":"left","bbox_2d":[100,200,800,900]}',
                        '```json\n{"action":"STOP","bbox_2d":[100,200,1200,1100]}\n```',
                        '{"action":"STOP"}', "unparseable response",
                    ):
                        for frame in (image, None):
                            expected = self.exercise(baseline.LaViRANavigationAPI, profile, reply,
                                                     frame, history, config, False)
                            actual = self.exercise(shared.LaViRANavigationAPI, profile, reply,
                                                   frame, history, config, True)
                            self.assertEqual(actual, expected)
                    for text in ('{"x":{"y":1}}', '```json\n{"action":"STOP"}\n```',
                                 "{turn_direction:'left',}", "not JSON",
                                 '{"reasoning": "near door", "turn_direction":"right" broken}',
                                 '{"bbox_2d":[1, 2, 3, 4],"stop":true broken}'):
                        self.assertEqual(shared.safe_json_loads(text), helpers.safe_json_loads(text))
                        self.assertEqual(compatibility_helpers.safe_json_loads(text), helpers.safe_json_loads(text))
                    if profile == "cobot":
                        for todo in ("", "[ ] one [ ] two", "[x] one\n[ ] two", "[x] done"):
                            for name in ("mark_first_incomplete_task_completed", "is_all_tasks_completed"):
                                self.assertEqual(getattr(compatibility_helpers, name)(todo),
                                                 getattr(helpers, name)(todo))
                    with patch.dict(sys.modules, {"config": types.SimpleNamespace(Config=config)}):
                        spec = importlib.util.spec_from_file_location(
                            profile + "_entry", ROOT / "real-world-code" / platform / "robot/navigation_api.py")
                        entry = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(entry)
                        self.assertEqual(self.exercise(entry.LaViRANavigationAPI, profile,
                                                       reply, image, history, config, False),
                                         self.exercise(shared.LaViRANavigationAPI, profile,
                                                       reply, image, history, config, True))

    def exercise(self, cls, profile, reply, image, history, config, shared):
        client = RecordingClient(reply)
        api = cls(client, config=config) if shared else cls(client)
        frames = [{"img_base64": "test-image", "label": "FRONT (0°)"},
                  {"img_base64": "", "label": "LEFT (90°)"}]
        results = [api.generate_initial_todo("find the chair", frames)]
        if profile == "cobot":
            results.append(api.decide_direction(frames, "find the chair", [], "[ ] chair"))
            results.append(api.query_llm_bbox(image, "find the chair", "near door", [],
                                             "chair", 1600, 1200))
        else:
            results.append(api.language_action("find the chair", "chair", "[ ] chair",
                                               history, frames, 2))
            kwargs = {"progress_analysis": "near door"} if profile == "go1" else {}
            results.append(api.vision_action(image, "find the chair", "chair", "chair",
                                             False, 2, **kwargs))
        return client.calls, results


if __name__ == "__main__":
    unittest.main()
