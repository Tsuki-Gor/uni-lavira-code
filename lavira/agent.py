"""One high-level task driver, with explicit release-compatibility choices."""

from threading import Event
import time

from navigation_runtime import PixelGoal, RelativeGoal
from .runtime import Progress, TaskResult, TaskState
from .reasoning._helpers import go1_img_to_base64, g1_img_to_base64


class LaViRAAgent:
    def __init__(self, reasoning, profile):
        if profile not in ("go1", "g1"):
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
        state = self.state = TaskState()
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
            if state.step == 1:
                if self.profile == "g1":
                    state.todo_list = self.reasoning.generate_initial_todo(task.instruction, frames)
                elif not state.todo_list:
                    state.todo_list = "No TODO list yet. Please generate one."

            progress("reasoning")
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
            runtime.turn(angles[direction])
            view = runtime.observe_current()
            if view is None:
                if self.profile == "g1":
                    state.step += 1
                continue

            kwargs = {}
            if self.profile == "go1":
                landmark = decision.get("expected_landmark", "")
                target_text = landmark if landmark and len(landmark) > 2 else ""
                kwargs["progress_analysis"] = decision.get("progress_analysis", "")
            else:
                target_text = "Go {}. {}".format(direction, decision.get("reasoning", ""))
            tactical = self.reasoning.vision_action(
                img_np=view.image, instruction=task.instruction,
                global_target=task.instruction, strategic_goal=target_text,
                strategic_stop=False, current_step=state.step, **kwargs)
            bbox = tactical.get("bbox_2d")
            valid_bbox = bool(bbox) and len(bbox) == 4
            stop = stop or tactical.get("action", "NAVIGATE") == "STOP"
            if stop and not valid_bbox:
                reason = "model_stop"
                break

            goal = RelativeGoal(1.5, 0.0)
            if valid_bbox:
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
            if stop:
                if self.profile == "go1":
                    time.sleep(3.0)
                reason = "model_stop"
                break
            state.step += 1
        else:
            if cancellation.is_set():
                reason = "cancelled"

        if self.profile == "go1":
            time.sleep(2.0)
        progress("finished")
        return TaskResult(reason, state, cancellation.is_set(), report)

    def _panorama_frames(self, panorama):
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
