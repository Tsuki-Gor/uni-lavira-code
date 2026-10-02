"""Compare simulator reasoning against the release without loading Habitat."""

import contextlib
import copy
import importlib
import importlib.util
import io
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASE = "sim-code/habitat/vlnce_baselines/"


def original(name, path, package=False):
    module = types.ModuleType(name)
    module.__package__ = name if package else name.rpartition(".")[0]
    if package:
        module.__path__ = []
    sys.modules[name] = module
    source = subprocess.check_output(
        ["git", "show", "215e7ac:" + BASE + path], cwd=ROOT, text=True)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


class Model:
    def __init__(self):
        self.calls = []
        self.replies = []

    def eval(self):
        self.calls.append(("eval",))

    def generate(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class HabitatReasoningRegressionTest(unittest.TestCase):
    def test_requests_history_state_and_outputs_match_release(self):
        self.assertTrue((ROOT / "lavira/reasoning/habitat.py").is_file(),
                        "Habitat reasoning still depends on the simulator package")
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("requires existing NumPy and Pillow dependencies")
        Image.init()
        log = logging.getLogger("habitat-reasoning-test")
        log.disabled = True
        model = Model()
        api = types.SimpleNamespace(LaViRA_API=lambda **kwargs: model)
        with patch.dict(sys.modules, {
            "habitat": types.SimpleNamespace(logger=log),
            "baseline": types.ModuleType("baseline"),
            "baseline.utils.api": api,
            "baseline.utils.visualization": types.SimpleNamespace(LaViRAVisualizer=object),
        }), patch.dict(os.environ, {"LA_API_KEY": "test", "VA_API_KEY": "test"}):
            prompts = types.ModuleType("baseline.prompts")
            prompts.__path__ = []
            sys.modules["baseline.prompts"] = prompts
            for task in ("vln", "objnav", "eqa"):
                part = original("baseline.prompts.prompts_" + task,
                                "prompts/prompts_" + task + ".py")
                setattr(prompts, "prompts_" + task, part)
            original("baseline.prompts", "prompts/__init__.py", package=True)
            baseline = original("baseline.agent", "agent.py")
            shared = importlib.import_module("lavira.reasoning.habitat")
            with patch.dict(sys.modules, {"habitat": None, "vlnce_baselines": None}):
                importlib.reload(shared)

            with tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()), patch("time.time", return_value=100.0):
                for task in ("VLN", "OBJNAV", "EQA"):
                    for case in ("normal", "history", "stop", "backtrack", "invalid",
                                 "malformed", "no_todo", "blocked"):
                        with self.subTest(task=task, case=case):
                            def legacy(client, recorder, **kwargs):
                                nonlocal model
                                model = client
                                return baseline.VLMReasoningAgent(recorder, **kwargs)

                            def core(client, recorder, **kwargs):
                                return shared.VLMReasoningAgent(
                                    client, image_height=recorder.height,
                                    on_bbox=recorder._save_rgb_with_bbox, log=log, **kwargs)

                            expected = self.exercise(legacy, task, case, directory, np, Image)
                            actual = self.exercise(core, task, case, directory, np, Image)
                            self.assertEqual(actual, expected)

                # The old evaluator import remains a real assembly entrypoint.
                spec = importlib.util.spec_from_file_location(
                    "baseline.entry", ROOT / BASE / "agent.py")
                entry = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(entry)
                model = Model()
                recorder = types.SimpleNamespace(height=12, _save_rgb_with_bbox=lambda *a, **kw: None)
                self.assertIsInstance(entry.VLMReasoningAgent(recorder), shared.VLMReasoningAgent)
                self.assertEqual(model.calls, [("eval",)])

    def exercise(self, make, task, case, directory, np, Image):
        model = Model()
        renders = []
        recorder = types.SimpleNamespace(height=10, _save_rgb_with_bbox=lambda rgb, bbox, **kw:
                                         renders.append((np.asarray(rgb).tolist(), copy.deepcopy(bbox), kw)))
        agent = make(model, recorder, task_type=task, use_guideline=True,
                     use_todo_list=case != "no_todo", allow_move_behind=False,
                     debug_logging=True, log_dir=directory)
        rgb = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        panorama = [{"rgb": np.roll(rgb, i, axis=0),
                     "depth": np.full((12, 16, 1), .05 if case == "blocked" else .5)}
                    for i in range(4)]
        visited = [{"step": i * 12, "panorama_frames": copy.deepcopy(panorama),
                    "world_coords": (i * 20, i * 10), "init_image": Image.fromarray(rgb),
                    "dir_image": Image.fromarray(np.roll(rgb, i, axis=1)),
                    "direction_decision": "left", "description": "chair"} for i in range(3)]
        # Distinct snapshots detect a change in which waypoint supplies each view.
        visited[0]["panorama_frames"][0]["rgb"] = rgb[::-1].copy()
        history = ([{"step": i, "image": Image.fromarray(np.roll(rgb, i, axis=1))}
                    for i in (0, 6, 12, 18, 24)] if case == "history" else None)
        if case == "no_todo":
            model.replies = ["chair"] if task == "VLN" else []
        else:
            model.replies = ['{"todos":[{"content":"Find chair","status":"pending"}],'
                             '"last_object":"chair"}']
        agent.generate_todo_list("find the chair", agent._get_initial_views(visited), "ep", 0)

        nav = '{"action":"navigate to left","progress_analysis":"near chair",' \
              '"reasoning":"seen","stair":"down",' \
              '"todo_updates":[{"index":0,"status":"completed","result":"chair seen"}]}'
        replies = {
            "stop": ['{"action":"stop"}'],
            "backtrack": ['{"action":"backtrack to waypoint 0","stop":true}'],
            "invalid": ["invalid"] * 6,
            "malformed": ["{not json}"],
        }
        model.replies = replies.get(case, [nav]).copy()
        decisions = [agent.navigate_or_backtrack("find the chair", visited, "try again", "ep", 24, history)]
        model.replies = (["invalid"] * 6 if case == "invalid" else
                         ['{"action":"STOP","bbox_2d":[1,2]}'] if case == "malformed" else
                         ['{"action":"NAVIGATE","bbox_2d":[-100,200,1100,900],"target":"chair"}'])
        decisions.append(agent.query_llm("find the chair", visited, rgb, panorama[0]["depth"],
                                        16, 12, 24, "near chair", "STOP", "ep"))
        model.replies = ["invalid" if case == "invalid" else '{"decision":"STOP","reasoning":"seen"}']
        decisions.append(agent.double_check_stop("find the chair", panorama, visited, "ep", 24, history))
        model.replies = ["invalid" if case == "invalid" else '{"action":"navigate to right","stair":"up"}']
        decisions.append(agent.replan_at_backtrack("find the chair", visited[:2], visited[2:], "ep", 24, history))
        if task == "EQA":
            model.replies = ([RuntimeError("offline model failure")] if case == "invalid" else
                             ["{bad}"] if case == "malformed" else ['{"answer":"red"}'])
            decisions.append(agent.query_llm_oracle(panorama, "What color is the chair?"))
        memory = [(item.get("description"), item.get("bbox"), item.get("reasoning")) for item in visited]
        state = (copy.deepcopy(agent.todo_list), agent.last_object, agent.stair,
                 agent.use_guideline, getattr(agent, "reasoning_todo", None))
        agent.reset()
        return model.calls, decisions, renders, memory, state, (agent.todo_list, agent.last_object, agent.stair)


if __name__ == "__main__":
    unittest.main()
