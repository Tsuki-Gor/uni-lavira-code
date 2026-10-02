"""Panoramic arm/camera capture and pixel navigation, independent of LaViRA."""

from math import isfinite
import os
import time

import cv2

from . import ExecutionReport, Frame, Panorama, PixelGoal


class GimbalRobotRuntime:
    def __init__(self, robot, session_root, include_depth=False):
        self.robot = robot
        self.session_root = session_root
        self.include_depth = include_depth
        self.reset_thread = None
        self.cycle_dir = None

    @property
    def running(self):
        # The existing driver checks cancellation/budget, not robot.running.
        return True

    def observe_panorama(self, step):
        self.cycle_dir = os.path.join(self.session_root, "cycle_{:02d}".format(step))
        os.makedirs(self.cycle_dir, exist_ok=True)
        images, depths, self.reset_thread = self.robot.collect_panoramic_images()
        self.reset_thread.start()
        directory = os.path.join(self.cycle_dir, "panorama")
        os.makedirs(directory, exist_ok=True)
        names = ("front", "left_front", "left", "left_back", "behind", "right_back", "right", "right_front")
        headings = (0, 45, 90, 135, 180, -135, -90, -45)
        views = {}
        for index, image in enumerate(images):
            if image is not None:
                path = os.path.join(directory, "pano_{}.jpg".format(index))
                cv2.imwrite(path, image)
                if index < len(names):
                    depth = depths[index] if index < len(depths) else None
                    views[names[index]] = Frame(image=image, path=path, depth=depth,
                                               heading_degrees=headings[index])
        return Panorama(views, record_dir=self.cycle_dir)

    def turn(self, degrees):
        if not isfinite(degrees):
            raise ValueError("turn angle must be finite degrees")
        self.robot.rotate_angle(degrees)

    def finish_observation(self):
        if self.reset_thread is not None and self.reset_thread.is_alive():
            self.reset_thread.join()

    def observe_current(self):
        time.sleep(2.0)
        self.finish_observation()
        rgb, depth = None, None
        for _ in range(10):
            with self.robot.image_lock:
                if self.robot.current_front_rgb is not None:
                    rgb = self.robot.current_front_rgb.copy()
                if self.include_depth and self.robot.current_front_depth is not None:
                    depth = self.robot.current_front_depth.copy()
            if rgb is not None:
                break
            time.sleep(0.2)
        if rgb is None:
            return None
        cv2.imwrite(os.path.join(self.cycle_dir, "raw_rgb.jpg"), rgb)
        return Frame(image=rgb, depth=depth, width=self.robot.rgb_width, height=self.robot.rgb_height)

    def navigate(self, goal, step):
        if not isinstance(goal, PixelGoal):
            raise TypeError("gimbal navigation requires a PixelGoal")
        arrived = self.robot.navigate_dynamic(goal.u, goal.v, cycle_dir=self.cycle_dir)
        return ExecutionReport(execution_called=True, arrived=arrived)
