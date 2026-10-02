"""One high-level task driver, with explicit release-compatibility choices."""

from threading import Event
import os
import time

from navigation_runtime import PixelGoal, RelativeGoal
from .runtime import Progress, TaskResult, TaskState
from .reasoning._helpers import (
    go1_img_to_base64, g1_img_to_base64, image_to_base64,
    mark_first_incomplete_task_completed, is_all_tasks_completed,
)


COBOT_VIEWS = (
    ("front", "FRONT (0°)"), ("left_front", "LEFT-FRONT (45°)"),
    ("left", "LEFT (90°)"), ("left_back", "LEFT-BACK (135°)"),
    ("behind", "BACK (180°) - NO IMAGE"), ("right_back", "RIGHT-BACK (-135°)"),
    ("right", "RIGHT (-90°)"), ("right_front", "RIGHT-FRONT (-45°)"),
)
COBOT_ANGLES = {"front": 0, "left_front": 45, "left": 90, "left_back": 135,
                "right_back": -135, "right": -90, "right_front": -45}


class LaViRAAgent:
    def __init__(self, reasoning, profile):
        if profile not in ("go1", "g1", "cobot"):
            raise ValueError("Unsupported navigation profile: " + profile)
        self.reasoning = reasoning
        self.profile = profile
        self.state = TaskState()

    def run(self, task, runtime, cancellation=None, on_progress=None):
        if task.kind not in ("vln", "object_nav"):
            raise ValueError("Task not migrated to the shared driver: " + task.kind)
        if task.max_cycles is not None and task.max_cycles < 0:
            raise ValueError("max_cycles must be non-negative or None")
        cancellation = cancellation if cancellation is not None else Event()
        cobot = self.profile == "cobot"
        state = self.state = TaskState(step=0 if cobot else 1)
        report = None
        reason = "environment_finished"

        def progress(stage):
            if on_progress is not None:
                on_progress(Progress(stage, state.step, state.cycles))

        progress("started")
        while runtime.running and not cancellation.is_set():
            if task.max_cycles is not None and state.cycles >= task.max_cycles:
                reason = "budget_exhausted"
                break
            state.cycles += 1
            progress("observing")
            panorama = runtime.observe_panorama(state.step)
            if panorama is None:
                reason = "observation_failed"
                break
            frames = self._panorama_frames(panorama)
            if state.step == 1 and not cobot:
                if self.profile == "g1":
                    state.todo_list = self.reasoning.generate_initial_todo(task.instruction, frames)
                elif not state.todo_list:
                    state.todo_list = "No TODO list yet. Please generate one."

            progress("reasoning")
            if cobot:
                decision = self.reasoning.decide_direction(
                    frames, task.instruction, state.visited_targets, state.todo_list,
                    cycle_dir=panorama.record_dir)
                state.todo_list = decision.get("updated_todo_list", state.todo_list)
                if decision.get("action") == "STOP":
                    runtime.finish_observation()
                    reason = "model_stop"
                    break
                direction = decision.get("turn_direction", "front")
                target_text = decision.get("expected_landmark", "target ahead")
                state.visited_targets.append({
                    "step": len(state.visited_targets) + 1, "direction": direction,
                    "target": target_text, "description": "Turned {} toward {}".format(direction, target_text)})
                angles, stop = COBOT_ANGLES, False
            else:
                decision = self.reasoning.language_action(
                    instruction=task.instruction, global_target=task.instruction,
                    todo_list=state.todo_list, iplanner_history=panorama.history,
                    panorama_frames=frames, current_step=state.step)
                if decision.get("updated_todo_list"):
                    state.todo_list = decision["updated_todo_list"]
                direction = decision.get("turn_direction", "front").lower()
                angles = {"front": 0, "left": 90, "right": -90}
                if self.profile == "g1":
                    angles["behind"] = -180
                    stop = decision.get("stop", False)
                    stop = stop.lower() == "true" if isinstance(stop, str) else bool(stop)
                else:
                    stop = decision.get("action", "NAVIGATE") == "STOP"
                if direction not in angles:
                    direction = "front"
            if direction in angles:
                runtime.turn(angles[direction])
            view = runtime.observe_current()
            if view is None:
                if self.profile in ("g1", "cobot"):
                    state.step += 1
                if cobot and not cancellation.is_set():
                    time.sleep(1.0)
                continue

            kwargs = {}
            if self.profile == "go1":
                landmark = decision.get("expected_landmark", "")
                target_text = landmark if landmark and len(landmark) > 2 else ""
                kwargs["progress_analysis"] = decision.get("progress_analysis", "")
            elif not cobot:
                target_text = "Go {}. {}".format(direction, decision.get("reasoning", ""))
            if cobot:
                tactical = self.reasoning.query_llm_bbox(
                    view.image, task.instruction, decision.get("progress_analysis", ""),
                    state.visited_targets, target_text, view.width, view.height,
                    cycle_dir=panorama.record_dir)
                bbox = tactical.get("bbox_2d", []) if tactical else None
            else:
                tactical = self.reasoning.vision_action(
                    img_np=view.image, instruction=task.instruction,
                    global_target=task.instruction, strategic_goal=target_text,
                    strategic_stop=False, current_step=state.step, **kwargs)
                bbox = tactical.get("bbox_2d")
            valid_bbox = bool(bbox) and len(bbox) == 4
            if not cobot:
                stop = stop or tactical.get("action", "NAVIGATE") == "STOP"
            if stop and not valid_bbox:
                reason = "model_stop"
                break

            goal = RelativeGoal(1.5, 0.0)
            if cobot:
                u, v = view.width / 2, view.height / 2
                if valid_bbox:
                    x1, y1, x2, y2 = [int(value) for value in bbox]
                    u, v = (x1 + x2) / 2, y2
                    self._record_bbox(view, (x1, y1, x2, y2), panorama.record_dir)
                goal = PixelGoal(view, u, v)
            elif valid_bbox:
                x1, y1, x2, y2 = bbox
                u = int((x1 + x2) / 2)
                if self.profile == "go1":
                    v = int(y2 - 0.25 * (y2 - y1))
                else:
                    height = view.depth.shape[0] if view.depth is not None else 480
                    v = min(int(y2), height - 1)
                goal = PixelGoal(view, u, v, fallback=goal)
            progress("executing")
            report = runtime.navigate(goal, state.step)
            if cobot:
                if report.arrived:
                    state.todo_list = mark_first_incomplete_task_completed(state.todo_list)
                if is_all_tasks_completed(state.todo_list):
                    reason = "todo_complete"
                    break
            if stop:
                if self.profile == "go1":
                    time.sleep(3.0)
                reason = "model_stop"
                break
            state.step += 1
            if cobot and not cancellation.is_set():
                time.sleep(1.0)
        else:
            if cancellation.is_set():
                reason = "cancelled"

        if self.profile == "go1":
            time.sleep(2.0)
        progress("finished")
        return TaskResult(reason, state, cancellation.is_set(), report)

    def _panorama_frames(self, panorama):
        if self.profile == "cobot":
            return [{"img_base64": image_to_base64(panorama.views[name].image), "label": label}
                    for name, label in COBOT_VIEWS if name in panorama.views]
        step = self.state.step
        unit = "°" if self.profile == "go1" else " deg"
        descriptions = (
            ("front", 360, "Image 1: The current FORWARD view (Step {}).".format(step)),
            ("right", 90, "Image 2: The view after turning 90{} to the RIGHT (Step {}).".format(unit, step)),
            ("behind", 180, "Image 3: The view directly BEHIND (180 deg turn) (Step {}).".format(step)),
            ("left", 270, "Image 4: The view after turning 90{} to the LEFT (Step {}).".format(unit, step)),
        )
        encode = go1_img_to_base64 if self.profile == "go1" else g1_img_to_base64
        return [{"angle": angle, "label": label, "img_base64": encode(panorama.views[name].path)}
                for name, angle, label in descriptions if name in panorama.views]

    @staticmethod
    def _record_bbox(view, bbox, record_dir):
        if not record_dir:
            return
        import cv2
        x1, y1, x2, y2 = bbox
        debug = view.image.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
        directory = os.path.join(record_dir, "bbox")
        os.makedirs(directory, exist_ok=True)
        cv2.imwrite(os.path.join(directory, "bbox.jpg"), debug)
