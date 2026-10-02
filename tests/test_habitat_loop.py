"""Execute the actual release rollout with offline environment/controller doubles."""

import ast
import copy
import hashlib
import json
import importlib
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from threading import Event
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = "sim-code/habitat/vlnce_baselines/ZS_Evaluator_mp.py"


def rollout_methods(original, namespace):
    source = (subprocess.check_output(["git", "show", "215e7ac:" + EVALUATOR], cwd=ROOT, text=True)
              if original else (ROOT / EVALUATOR).read_text())
    tree = ast.parse(source)
    if not original:
        host = ROOT / "sim-code/habitat/vlnce_baselines/habitat_environment.py"
        tree.body.extend(ast.parse(host.read_text()).body)
        from lavira.agent import LaViRAAgent
        from lavira.runtime import Task
        from navigation_runtime.mapped_simulator import MappedSimulatorRuntime
        namespace.update(LaViRAAgent=LaViRAAgent, Task=Task, MappedSimulatorRuntime=MappedSimulatorRuntime)
    methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name in
               ("rollout", "get_panorama", "_save_eqa_metric")]
    # Only platform imports/constructors are excluded. These method bodies run
    # unchanged, including panorama actions, decisions and map updates.
    exec(compile(ast.Module(body=methods, type_ignores=[]), EVALUATOR, "exec"), namespace)
    return {node.name: namespace[node.name] for node in methods}


class HabitatLoopTest(unittest.TestCase):
    def test_original_and_current_rollout_traces(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("requires existing NumPy and Pillow dependencies")
        Image.init()
        from lavira.agent import LaViRAAgent
        LaViRAAgent(object(), "habitat")
        with tempfile.TemporaryDirectory() as directory:
            for task in ("VLN", "OBJNAV", "EQA"):
                cases = ("normal", "left", "right", "behind", "stop", "reject_stop",
                         "backtrack_only", "backtrack_replan", "backtrack_alias_pose", "timeout", "budget",
                         "panorama_done", "environment_done", "navdp", "navdp_cache", "no_history",
                         "invalid_backtrack", "blocked_projection", "zero_budget", "negative_budget")
                if task == "EQA":
                    cases += ("qa_failure", "already_scored")
                for case in cases:
                    with self.subTest(task=task, case=case):
                        expected = self.capture(True, task, case, directory, np, Image)
                        actual = self.capture(False, task, case, directory, np, Image)
                        self.assertEqual(actual, expected)

    def test_cancel_and_replacement_algorithm_use_neutral_runtime(self):
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("requires existing image dependencies")
        Image.init()
        with tempfile.TemporaryDirectory() as directory:
            self.capture(False, "VLN", "cancel", directory, np, Image)
            self.capture(False, "VLN", "replacement_algorithm", directory, np, Image)

    def capture(self, original, task_type, case, directory, np, Image):
        trace = []
        cancellation = Event()
        progress = []

        def canonical(value):
            if isinstance(value, np.ndarray):
                if value.ndim < 2:
                    return value.tolist()
                return (value.shape, str(value.dtype), hashlib.sha256(value.tobytes()).hexdigest())
            if isinstance(value, Image.Image):
                return (value.size, hashlib.sha256(value.tobytes()).hexdigest())
            if isinstance(value, dict):
                # _views binds historical pose/calibration to the same images;
                # model messages still use the original panorama_frames fields.
                return {key: canonical(item) for key, item in value.items() if key != "_views"}
            if isinstance(value, (tuple, list)):
                return [canonical(item) for item in value]
            return value

        class Tensor:
            def __init__(self, array):
                self.array = array

            def float(self):
                return self

            def to(self, device):
                return self

        torch = types.SimpleNamespace(Tensor=Tensor, from_numpy=Tensor)
        actions = types.SimpleNamespace(MOVE_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3, STOP=0)
        log = logging.getLogger("habitat-loop-test")
        log.disabled = True

        class Environment:
            pose = np.array([2., 2., 0.])
            count = 0

            def current_episodes(self):
                return [types.SimpleNamespace(episode_id="ep", scene_id="scene",
                        question=types.SimpleNamespace(question_text="find the chair", answer_text="red"))]

            def call_at(self, index, name):
                return types.SimpleNamespace(_sim=types.SimpleNamespace(
                    get_agent_state=lambda: types.SimpleNamespace(position=self.pose.copy())))

            def observation(self):
                return {"rgb": np.full((12, 16, 3), self.count % 256, dtype=np.uint8),
                        "depth": np.full((12, 16, 1), .5, dtype=np.float32),
                        "sensor_pose": self.pose.copy()}

            def step(self, requests):
                action = requests[0]["action"]
                trace.append(("action", action))
                self.count += 1
                self.pose = self.pose.copy()
                if action == 2:
                    self.pose[2] = (self.pose[2] + 30) % 360
                elif action == 3:
                    self.pose[2] = (self.pose[2] - 30) % 360
                elif action == 1 and case != "timeout":
                    angle = np.deg2rad(self.pose[2])
                    distance = .1 if case == "navdp_cache" else 1.
                    self.pose[:2] += distance * np.array([np.cos(angle), np.sin(angle)])
                done = ((action == 0 and task_type != "EQA") or
                        (case == "panorama_done" and self.count == 3) or
                        (case == "environment_done" and self.count == 14))
                return [(self.observation(), 0., done, {"steps_taken": self.count})]

        env = Environment()

        class Mapping:
            one_step_full_map = types.SimpleNamespace(fill_=lambda x: None)
            one_step_local_map = types.SimpleNamespace(fill_=lambda x: None)
            map_shape = (160, 160)
            shared_pose = np.array([env.pose.copy()])

            def __call__(self, observations, poses, current_step):
                hint = host.agent.stair if original else getattr(host, "terrain_hint", False)
                trace.append(("mapping_input", current_step, hint))

            def get_full_map(self):
                return np.zeros((1, 4, 160, 160))

            def update_map(self, step, detected, episode):
                trace.append(("map", step))
                self.shared_pose[:] = env.pose
                pose = self.shared_pose if case == "backtrack_alias_pose" else np.array([env.pose.copy()])
                return self.get_full_map(), pose, self.get_full_map()

        class Reasoning:
            logger = log
            stair = False
            use_todo_list = True
            todo_list = [{"content": "chair", "status": "pending", "result": ""}]
            todo_verification_feedback = ""
            backtrack_second_chance = case != "backtrack_only"
            decisions = 0
            checks = 0

            def reset(self):
                pass

            def _format_todo_for_prompt(self):
                return "todo"

            def navigate_or_backtrack(self, **kwargs):
                trace.append(("la", canonical(kwargs)))
                self.decisions += 1
                if case == "cancel":
                    cancellation.set()
                if case == "invalid_backtrack" and self.decisions == 2:
                    return {"action": "BACKTRACK", "waypoint": 999}
                if case.startswith("backtrack") and self.decisions == 2:
                    return {"action": "BACKTRACK", "waypoint": 0}
                return {"action": "NAVIGATE", "direction": case if case in ("left", "right", "behind") else "forward",
                        "stop_signal": case in ("stop", "reject_stop") or self.decisions >= 3,
                        "progress_analysis": "near", "reasoning": "seen"}

            def query_llm(self, **kwargs):
                trace.append(("va", canonical(kwargs)))
                result = {"x1": 2, "y1": 3, "x2": 10, "y2": 9,
                          "target": "chair", "reasoning": "seen", "progress": "near"}
                kwargs["visited_targets"][-1].update(description="chair", bbox=copy.deepcopy(result))
                return result

            def double_check_stop(self, instruction, panorama, visited, **kwargs):
                trace.append(("stop_check", instruction, canonical(panorama), canonical(visited), canonical(kwargs)))
                self.checks += 1
                return case != "reject_stop", {"decision": "CONTINUE" if case == "reject_stop" else "STOP"}

            def replan_at_backtrack(self, instruction, visited, failed, **kwargs):
                trace.append(("replan", instruction, canonical(visited), canonical(failed), canonical(kwargs)))
                self.stair = "down"
                return "left", {"action": "navigate to left", "stair": "down"}

            def query_llm_oracle(self, panorama, instruction):
                trace.append(("qa", canonical(panorama), instruction))
                if case == "qa_failure":
                    raise RuntimeError("offline QA failure")
                return "red"

        def project(**kwargs):
            trace.append(("project", canonical(kwargs)))
            pose = kwargs["full_pose"]
            angle = np.deg2rad(pose[2])
            return (pose[0] + 2 * np.cos(angle), pose[1] + 2 * np.sin(angle))

        def plan(pose, goal, *args):
            trace.append(("fmm", canonical(pose), canonical(goal), args[3]))
            delta = goal[::-1] * .05 - pose[:2]
            angle = (np.rad2deg(np.arctan2(delta[1], delta[0])) - pose[2] + 180) % 360 - 180
            return 2 if angle > 15 else 3 if angle < -15 else 1

        def navdp(goals, images, depths):
            trace.append(("navdp", canonical(goals), canonical(images), canonical(depths)))
            x, y = goals[0, :2]
            return np.array([[[x / 2, y / 2], [x, y]]]), None, None, None

        namespace = {"np": np, "Image": Image, "torch": torch, "logger": log, "os": os, "json": json,
                     "Observations": dict, "List": list, "Dict": dict, "HabitatSimActions": actions,
                     "get_world_xz_from_pixel": project,
                     "calculate_displacement": lambda a, b, resolution: np.linalg.norm(a[:2] - b[:2]) / .05,
                     "collision_check_fmm": lambda *args: np.zeros((160, 160))}
        methods = rollout_methods(original, namespace)
        host = types.SimpleNamespace(
            envs=env, agent=Reasoning(), mapping_module=Mapping(), policy=types.SimpleNamespace(_get_action=plan),
            navdp_agent=types.SimpleNamespace(reset_env=lambda n: trace.append(("navdp_reset",)), step_pointgoal=navdp),
            task_type=task_type, use_fmm=not case.startswith("navdp"),
            use_navdp=case.startswith("navdp"), use_iplanner=False,
            instruction="find the chair", destination="goal", current_episode_id="ep", current_step=0,
            width=16, height=12, resolution=5, map_shape=(160, 160), target_reached_threshold=15,
            use_continuous_history=case != "no_history", history_interval=2, history_images=[], visited_targets=[],
            total_backtracks=2, total_waypoints=5, pos_list=[], detected_classes=[],
            max_step={"budget": 1, "already_scored": 1, "timeout": 50,
                      "zero_budget": 0, "negative_budget": -1}.get(case, 200),
            traversable=np.zeros((160, 160)) if case == "blocked_projection" else np.ones((160, 160)),
            collision_map=np.zeros((160, 160)),
            visualize=False, device="cpu", save_dir=directory, local_rank=0, world_size=1,
            state_eps={"ep": {"success": 1.0}} if case == "already_scored" else {},
            last_submitted_answer=None, _eqa_done_requested=False,
            config=types.SimpleNamespace(NUM_ENVIRONMENTS=1, EVAL_CKPT_PATH_DIR=directory,
                TASK_CONFIG=types.SimpleNamespace(DATASET=types.SimpleNamespace(SPLIT="test"),
                    SIMULATOR=types.SimpleNamespace(DEPTH_SENSOR=types.SimpleNamespace(MIN_DEPTH=.1, MAX_DEPTH=5.)))),
        )
        host.visualizer = types.SimpleNamespace(
            _save_depth=lambda *args, **kw: None, sync=lambda *args: None,
            _save_rgb_frame=lambda obs, step, *args, **kw: trace.append(("frame", step)),
            _save_waypoint_panorama_rgb=lambda pano, waypoint, step: trace.append(("waypoint", waypoint, step)))
        host._maps_initialization = lambda: (trace.append(("reset",)) or [env.observation()],
                                           host.mapping_module.shared_pose if case == "backtrack_alias_pose"
                                           else np.array([env.pose.copy()]))
        host._batch_obs = lambda obs: obs
        host._preprocess_depth = lambda image, *args: image[..., 0] * 100
        host._get_camera_intrinsics = lambda: np.array([[8., 0, 8.], [0, 6., 6.], [0, 0, 1.]])
        host._process_map = lambda *args: (np.ones((160, 160)), None, None)
        host._process_one_step_floor = lambda *args: None
        for name, method in methods.items():
            setattr(host, name, types.MethodType(method, host))

        def metric(infos, submitted_answer=None, is_timeout=False):
            if task_type == "EQA":
                host._save_eqa_metric(infos, submitted_answer, is_timeout)
            trace.append(("metric", copy.deepcopy(infos), is_timeout, copy.deepcopy(host.state_eps)))

        host._calculate_metric = metric
        if case == "replacement_algorithm":
            del host.agent
            module = importlib.import_module("navigation_runtime.mapped_simulator")
            with patch.dict(sys.modules, {"lavira": None, "habitat": None, "torch": None}):
                importlib.reload(module)
                from navigation_runtime import PixelGoal, MapGoal
                obs, pose = host._maps_initialization()
                runtime = module.MappedSimulatorRuntime(
                    host, obs, pose, tensor_api=torch, actions=actions, project_pixel=project,
                    displacement=namespace['calculate_displacement'],
                    collision_check=namespace['collision_check_fmm'], log=log)
                runtime.prepare_step(None, {})
                panorama = runtime.observe_panorama(0)
                view = panorama.views['left']
                self.assertEqual(view.color_order, 'RGB')
                self.assertEqual(view.depth_unit, 'normalized_0_1')
                self.assertEqual(view.pose_heading_offset, 90)
                goal = runtime.project(PixelGoal(view, 4, 9))
                self.assertEqual(goal.metres_per_cell, .05)
                with self.assertRaises(ValueError):
                    runtime.queue_navigation(MapGoal(1, 2, .05, 'another_map'))
                runtime.queue_navigation(goal, initial=True)
                self.assertTrue(runtime.execute_pending().execution_called)
                self.assertFalse(any(item[0] in ('la', 'va', 'qa') for item in trace))
            return
        if original:
            host.rollout()
        else:
            host.rollout(cancellation=cancellation, on_progress=progress.append)
            self.assertIsNone(host.result.physical_stop_confirmed)
            self.assertEqual(progress[0].stage, 'started')
            self.assertEqual(progress[-1].stage, 'finished')
            self.assertEqual(sum(item[0] == 'reset' for item in trace), 1)
            if case == 'cancel':
                self.assertEqual(host.result.reason, 'cancelled')
                self.assertTrue(host.result.cancellation_requested)
                self.assertEqual(host.result.state.step, 12)
                self.assertEqual(host.result.state.cycles, 1)
                self.assertFalse(any(item[0] == 'va' for item in trace))
            if task_type == 'EQA':
                no_answer = ('panorama_done', 'qa_failure', 'already_scored', 'zero_budget', 'negative_budget')
                self.assertEqual(host.result.answer, None if case in no_answer else 'red')
                if case == 'qa_failure':
                    self.assertEqual(host.result.answer_error, 'offline QA failure')
        self.assertTrue(any(item[0] == "metric" for item in trace))
        if case == "panorama_done":
            self.assertEqual(sum(item[0] == "action" for item in trace), 3)
            self.assertTrue(trace[-1][2])  # Preserve old timeout labeling on early panorama termination.
        if case == "reject_stop":
            self.assertEqual(host.agent.checks, 3)
        return trace, canonical(host.visited_targets), host.total_waypoints, host.total_backtracks


if __name__ == "__main__":
    unittest.main()
