"""
Stateless helpers for the Cobot Magic LaViRA iPlanner ObjectNav task.

Physical helpers do not load the navigation algorithm. Encoding, parsing
and TODO compatibility functions import shared LaViRA helpers only when called.

Public API
----------
TrajectoryFollower    – pure-pursuit velocity follower (reads limits from Config)
image_to_base64       – numpy/PIL -> resized JPEG base64 (max 640, q85)
pixel_to_robot_goal   – project a pixel + depth into a robot-frame (x, y) goal
mark_first_incomplete_task_completed – mark first ``[ ]`` TODO item done
is_all_tasks_completed               – True when no ``[ ]`` remains
save_debug_image      – overlay an iPlanner trajectory on an RGB frame
print_step / print_action / print_info / print_warning / print_error /
print_success         – coloured stdout logging
safe_json_loads       – robust JSON extraction from an LLM response
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from config import Config

try:
    from colorama import Fore, init as _colorama_init
    _colorama_init(autoreset=True)
except ImportError:  # pragma: no cover - optional dependency
    class _NoColour:
        RESET = CYAN = GREEN = BLUE = YELLOW = RED = MAGENTA = ""

        def __getattr__(self, _name: str) -> str:
            return ""

    Fore = _NoColour()


# ---------------------------------------------------------------------------
# Coloured stdout logging
# ---------------------------------------------------------------------------

def print_step(step_num: int, description: str) -> None:
    """Print step information."""
    print(Fore.CYAN + f"\n[STEP {step_num}] {description}")


def print_action(action: str, details: str = "") -> None:
    """Print action information."""
    print(Fore.GREEN + f"[ACTION] {action}" + (f" - {details}" if details else ""))


def print_info(info: str) -> None:
    """Print general information."""
    print(Fore.BLUE + f"[INFO] {info}")


def print_warning(warning: str) -> None:
    """Print warning information."""
    print(Fore.YELLOW + f"[WARNING] {warning}")


def print_error(error: str) -> None:
    """Print error information."""
    print(Fore.RED + f"[ERROR] {error}")


def print_success(success: str) -> None:
    """Print success information."""
    print(Fore.GREEN + f"[SUCCESS] {success}")


# ---------------------------------------------------------------------------
# Image / base64 helpers
# ---------------------------------------------------------------------------

def image_to_base64(image: Union[np.ndarray, "Image.Image"], max_dim: int = 640) -> str:
    """Compatibility entrypoint for the shared algorithm helper."""
    from lavira.reasoning._helpers import image_to_base64 as shared
    return shared(image, max_dim)


# ---------------------------------------------------------------------------
# Trajectory follower (pure-pursuit)
# ---------------------------------------------------------------------------

class TrajectoryFollower:
    """Pure-math pure-pursuit follower. Velocity limits come from ``Config``."""

    def __init__(self) -> None:
        self.DT = Config.FOLLOWER_DT
        self.MAX_V = Config.FOLLOWER_MAX_V
        self.MAX_W = Config.FOLLOWER_MAX_W
        self.MIN_V = Config.FOLLOWER_MIN_V
        self.MIN_W = Config.FOLLOWER_MIN_W
        self.ALPHA = Config.FOLLOWER_ALPHA
        self.LOOKAHEAD_DIST = Config.FOLLOWER_LOOKAHEAD_DIST
        self.last_v = 0.0
        self.last_w = 0.0

    def compute_velocity(
        self, trajectory_2d: Optional[Sequence[Sequence[float]]]
    ) -> Tuple[float, float]:
        """Return ``(v_cmd, w_cmd)`` for a robot-frame 2D trajectory."""
        if trajectory_2d is None or len(trajectory_2d) < 2:
            return 0.0, 0.0

        target_point = trajectory_2d[-1]
        for point in trajectory_2d:
            dist = math.hypot(point[0], point[1])
            if dist > self.LOOKAHEAD_DIST:
                target_point = point
                break

        tx, ty = target_point[0], target_point[1]
        dist_to_target = math.hypot(tx, ty)
        expected_time = max(dist_to_target / self.MAX_V, 0.1)

        v_cmd = tx / expected_time
        angle = math.atan2(ty, tx)
        w_cmd = angle / expected_time

        if abs(v_cmd) > 0.01 and abs(v_cmd) < self.MIN_V:
            v_cmd = self.MIN_V if v_cmd > 0 else -self.MIN_V
        if abs(w_cmd) > 0.01 and abs(w_cmd) < self.MIN_W:
            w_cmd = self.MIN_W if w_cmd > 0 else -self.MIN_W

        v_cmd = float(np.clip(v_cmd, -self.MAX_V, self.MAX_V))
        w_cmd = float(np.clip(w_cmd, -self.MAX_W, self.MAX_W))
        v_cmd = v_cmd * self.ALPHA + self.last_v * (1 - self.ALPHA)
        w_cmd = w_cmd * self.ALPHA + self.last_w * (1 - self.ALPHA)

        self.last_v = v_cmd
        self.last_w = w_cmd
        return v_cmd, w_cmd


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def pixel_to_robot_goal(
    u: float,
    v: float,
    depth_image: np.ndarray,
    depth_cx: float,
    depth_fx: float,
    window_size: int = 5,
) -> Tuple[float, float]:
    """Project pixel ``(u, v)`` plus depth into a robot-frame ``(x, y)`` goal.

    Samples the median valid depth in a small window around the pixel. Returns
    ``(Z, -x_cam)`` where ``Z`` is forward distance and ``-x_cam`` is the lateral
    offset in the robot frame. Falls back to ``(1.0, 0.0)`` if depth is missing.
    """
    if depth_image is None:
        return 1.0, 0.0

    h, w = depth_image.shape
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    patch = depth_image[
        max(0, v - window_size):min(h, v + window_size),
        max(0, u - window_size):min(w, u + window_size),
    ]
    valid = patch[patch > 0.1]
    Z = float(np.median(valid)) if len(valid) > 0 else 1.0
    x_cam = (u - depth_cx) * Z / depth_fx
    return Z, -x_cam


# ---------------------------------------------------------------------------
# TODO-list parsers
# ---------------------------------------------------------------------------

def mark_first_incomplete_task_completed(todo_list: str) -> str:
    """Compatibility entrypoint for the shared algorithm helper."""
    from lavira.reasoning._helpers import mark_first_incomplete_task_completed as shared
    return shared(todo_list)


def is_all_tasks_completed(todo_list: str) -> bool:
    """Compatibility entrypoint for the shared algorithm helper."""
    from lavira.reasoning._helpers import is_all_tasks_completed as shared
    return shared(todo_list)


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def save_debug_image(
    rgb: np.ndarray,
    path: Sequence[Sequence[float]],
    goal: Sequence[float],
    target_pixel_v: Optional[float],
    out_dir: str,
    intrinsics: Optional[List[List[float]]] = None,
) -> None:
    """Overlay an iPlanner trajectory + goal marker on ``rgb`` and save it.

    Projects robot-frame path points and the goal into image pixels using the
    camera intrinsics (defaults to ``Config.NAVDP_INTRINSICS``), draws the
    trajectory as a green polyline with sampled dots and the goal as a blue dot,
    then writes ``nav_debug_<ts>.jpg`` into ``out_dir/iplanner``. Errors are
    swallowed so debug output never breaks navigation.
    """
    try:
        if intrinsics is None:
            intrinsics = Config.NAVDP_INTRINSICS
        debug_img = rgb.copy()
        h, w, _ = debug_img.shape
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]

        if goal[0] > 0.1:
            u_g = int(cx - (fx * goal[1] / goal[0]))
            v_g = int(target_pixel_v) if target_pixel_v else int(cy + fy * 0.3 / goal[0])
            cv2.circle(debug_img, (u_g, v_g), 8, (255, 0, 0), -1)

        points_2d = []
        for pt in path:
            if pt[0] > 0.1:
                u = int(cx - fx * pt[1] / pt[0])
                v = int(cy + fy * 0.3 / pt[0])
                points_2d.append((u, v))

        if len(points_2d) > 1:
            for i in range(len(points_2d) - 1):
                p1, p2 = points_2d[i], points_2d[i + 1]
                if abs(p1[0] - p2[0]) < w / 2 and abs(p1[1] - p2[1]) < h / 2:
                    cv2.line(debug_img, p1, p2, (0, 255, 0), 2)
        for pt in points_2d[::5]:
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(debug_img, pt, 3, (0, 255, 0), -1)

        iplanner_dir = os.path.join(out_dir, "iplanner")
        os.makedirs(iplanner_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(iplanner_dir, f"nav_debug_{time.time():.1f}.jpg"),
            debug_img,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def safe_json_loads(text: str) -> Dict[str, Any]:
    """Compatibility entrypoint for the shared algorithm helper."""
    from lavira.reasoning._helpers import go1_safe_json_loads as shared
    return shared(text)
