"""Check captured-frame units and dimensions without constructing hardware."""

import tempfile
from threading import Lock, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class FrameMetadataTest(unittest.TestCase):
    def test_camera_frames_describe_raw_millimetres_and_image_dimensions(self):
        try:
            import numpy as np
            from navigation_runtime.camera_robot import CameraRobotRuntime
        except ImportError:
            self.skipTest("requires existing image dependencies")
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        depth = np.full((6, 8), 2500, dtype=np.uint16)
        robot = SimpleNamespace(
            capture_current_image=lambda: True, current_direction="front",
            direction_images={"front": {"rgb": image, "depth": depth}},
            camera_data={"front": {"rgb_image": image}, "right": {"rgb_image": None}},
            update_camera_data=lambda: None, rgb_image=image,
        )
        for delay in (None, 0.5):
            with self.subTest(recapture_delay=delay), patch("navigation_runtime.camera_robot.time.sleep"):
                runtime = CameraRobotRuntime(robot, "", (), 1.5, recapture_delay=delay)
                frame = runtime.observe_current()
                self.assertEqual(frame.depth_unit, "mm")
                self.assertEqual((frame.width, frame.height), (8, 6))
                self.assertIs(frame.image, image)
                self.assertIs(frame.depth, depth)
                self.assertEqual(int(frame.depth[0, 0]), 2500)
                name, current = next(runtime.observe_views())
                self.assertEqual((name, current.width, current.height), ("front", 8, 6))
                self.assertIs(current.image, image)
        runtime.live_cameras = (("front", "front"), ("right", "right"))
        views = dict(runtime.observe_views())
        self.assertEqual((views["front"].width, views["front"].height), (8, 6))
        self.assertEqual((views["right"].width, views["right"].height), (None, None))
        self.assertIsNone(views["right"].image)

    def test_gimbal_frames_keep_metre_depth_and_describe_available_views(self):
        try:
            import numpy as np
            from navigation_runtime.gimbal_robot import GimbalRobotRuntime
        except ImportError:
            self.skipTest("requires existing image dependencies")
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        side = np.zeros((3, 4, 3), dtype=np.uint8)
        depth = np.full((6, 8), 2.5, dtype=np.float32)
        robot = SimpleNamespace(
            collect_panoramic_images=lambda: (
                [image, side] + [None] * 6, [depth] + [None] * 7, Thread(target=lambda: None)),
            image_lock=Lock(), current_front_rgb=image, current_front_depth=depth,
            current_left_rgb=side, current_right_rgb=None, rgb_width=8, rgb_height=6,
        )
        with tempfile.TemporaryDirectory() as directory, patch("navigation_runtime.gimbal_robot.time.sleep"):
            runtime = GimbalRobotRuntime(robot, directory, include_depth=True)
            panorama = runtime.observe_panorama(0)
            front, left = panorama.views["front"], panorama.views["left_front"]
            self.assertEqual((front.depth_unit, front.width, front.height), ("m", 8, 6))
            self.assertIs(front.image, image)
            self.assertIs(front.depth, depth)
            self.assertEqual((left.width, left.height), (4, 3))
            self.assertIsNone(left.depth)
            current = runtime.observe_current()
            self.assertEqual((current.depth_unit, current.width, current.height), ("m", 8, 6))
            np.testing.assert_array_equal(current.depth, depth)
            np.testing.assert_array_equal(current.image, image)
            runtime.include_depth = False
            self.assertIsNone(runtime.observe_current().depth)
            views = dict(runtime.observe_views())
            self.assertEqual((views["front"].width, views["front"].height), (8, 6))
            self.assertEqual((views["left"].width, views["left"].height), (4, 3))
            self.assertIs(views["left"].image, side)
            self.assertEqual((views["right"].width, views["right"].height), (None, None))
            self.assertIsNone(views["right"].image)


if __name__ == "__main__":
    unittest.main()
