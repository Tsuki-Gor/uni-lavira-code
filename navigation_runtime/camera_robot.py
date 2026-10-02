"""Existing camera/iPlanner execution for Go1 and G1, usable by any algorithm."""

import os
import time
from math import isfinite

import numpy as np

from . import ExecutionReport, Frame, Panorama, PixelGoal, RelativeGoal


class CameraRobotRuntime:
    def __init__(self, robot, panorama_dir, directions, safety_distance,
                 require_panorama_success=False, recapture_delay=None, live_cameras=()):
        self.robot = robot
        self.panorama_dir = panorama_dir
        self.directions = directions
        self.safety_distance = safety_distance
        self.require_panorama_success = require_panorama_success
        self.recapture_delay = recapture_delay
        self.live_cameras = live_cameras

    @property
    def running(self):
        return self.robot.running

    def observe_panorama(self, step):
        captured = self.robot.capture_all_directions(step)
        if self.require_panorama_success and not captured:
            return None
        views = {}
        for name, file_angle, heading in self.directions:
            path = os.path.join(self.panorama_dir, "step{}".format(step), "view_{}.png".format(file_angle))
            if self.require_panorama_success or os.path.exists(path):
                views[name] = Frame(path=path, heading_degrees=heading)
        if not views:
            return None
        return Panorama(views, self.robot.iplanner_history)

    def turn(self, degrees):
        if not isfinite(degrees):
            raise ValueError("turn angle must be finite degrees")
        if degrees > 0:
            self.robot.rotate_left(degrees)
        elif degrees < 0:
            self.robot.rotate_right(-degrees)

    def observe_current(self):
        if not self.robot.capture_current_image():
            return None
        if self.recapture_delay is not None:
            time.sleep(self.recapture_delay)
            # Preserve the old second capture, including its unchecked return.
            self.robot.capture_current_image()
        data = self.robot.direction_images[self.robot.current_direction]
        height, width = data["rgb"].shape[:2]
        return Frame(image=data["rgb"], depth=data["depth"], depth_unit="mm",
                     width=width, height=height)

    def finish_observation(self):
        """This camera runtime has no pending scan operation."""

    def observe_views(self):
        """Read fixed camera views in host-specified order without rotating."""
        if self.live_cameras:
            frames = ((name, self.robot.camera_data[camera]["rgb_image"])
                      for name, camera in self.live_cameras)
        else:
            self.robot.update_camera_data()
            frames = (("front", self.robot.rgb_image),)
        for name, image in frames:
            height, width = image.shape[:2] if image is not None else (None, None)
            yield name, Frame(image=image, width=width, height=height)

    def navigate(self, goal, step, on_trajectory_image=None):
        record_options = {} if on_trajectory_image is None else {"on_trajectory_image": on_trajectory_image}
        target_pixel = None
        if isinstance(goal, PixelGoal):
            target_pixel = (goal.u, goal.v)
            fallback = goal.fallback
            goal_2d = (fallback.x, fallback.y) if fallback is not None else None
            # Compatibility: the original controllers read live depth and
            # intrinsics here. Switching to the captured frame is a policy change.
            depth = self.robot.get_depth_value(goal.u, goal.v, self.robot.current_direction)
            if depth is not None:
                camera = self.robot.camera_data[self.robot.current_camera]
                fx, cx = camera.get("fx"), camera.get("cx")
                if fx and cx:
                    goal_2d = (depth, -(goal.u - cx) * depth / fx)
            if goal_2d is None:
                return ExecutionReport(detail="projection_unavailable")
        elif isinstance(goal, RelativeGoal):
            goal_2d = (goal.x, goal.y)
        else:
            raise TypeError("camera navigation requires PixelGoal or RelativeGoal")

        data = self.robot.direction_images[self.robot.current_direction]
        rgb, depth = data["rgb"], data["depth"]
        trajectory, fear = self.robot.planner_client.get_plan(rgb, depth, goal_2d)
        if trajectory is not None:
            name = "step{}_plan_{}.jpg".format(step, time.strftime("%H%M%S"))
            self.robot.project_trajectory_to_image(
                trajectory, rgb, save_name=name, target_pixel=target_pixel,
                target_3d=(goal_2d[0], goal_2d[1], 0.0), **record_options)
        if trajectory is None or len(trajectory) == 0:
            return ExecutionReport(detail="planner_failed")

        try:
            path = np.array(trajectory)
            distances = np.linalg.norm((path[1:] - path[:-1])[:, :2], axis=1)
            target_distance = np.sum(distances) - self.safety_distance
            if target_distance <= 0:
                return ExecutionReport(detail="within_safety_buffer")
            current_distance = 0
            cutoff = len(path) - 1
            for index, distance in enumerate(distances):
                current_distance += distance
                if current_distance >= target_distance:
                    cutoff = index + 1
                    break
            truncated = trajectory[:cutoff + 1]
            if len(truncated) > 0:
                final = truncated[-1]
                self.robot.execute_trajectory(truncated, goal_local=(final[0], final[1]), **record_options)
            else:
                self.robot.execute_trajectory(trajectory, goal_local=goal_2d, **record_options)
        except Exception:
            # Existing fallback includes execution exceptions; changing this
            # possible replay requires a separate behavioral decision.
            self.robot.execute_trajectory(trajectory, goal_local=goal_2d, **record_options)
        return ExecutionReport(execution_called=True)
