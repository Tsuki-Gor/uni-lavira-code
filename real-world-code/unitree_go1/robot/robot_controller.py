"""
Unitree Go1 robot controller.

Wraps the Unitree High-level SDK plus four ROS RGB-D cameras and exposes
panorama capture, base rotation, depth lookup, and iPlanner-driven trajectory
execution.

Import safety
-------------
ROS (``rospy``), OpenCV (``cv2``), and the native Unitree SDK
(``robot_interface``) are imported lazily inside ``__init__`` / methods so this
module can be imported on a machine without ROS or the SDK installed
(``import robot.robot_controller`` succeeds with no hardware present).

Paths and network endpoints
---------------------------
The Unitree SDK directory comes from ``Config.UNITREE_SDK_PATH`` and the UDP
host / ports from ``Config.UNITREE_HOST`` / ``Config.UNITREE_LOCAL_PORT`` /
``Config.UNITREE_REMOTE_PORT``. The module ships no absolute paths.
"""
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from utils import print_action, print_error, print_info, print_success, print_warning
from config import Config
from robot.iplanner_client import IPlannerRemoteClient


def _import_unitree_sdk():
    """Import the Unitree High-level SDK, optionally extending ``sys.path``.

    Returns the imported module, or ``None`` if the SDK is unavailable.
    """
    if Config.UNITREE_SDK_PATH:
        sys.path.append(Config.UNITREE_SDK_PATH)
    try:
        import robot_interface as sdk  # noqa: WPS433 (runtime import is intentional)
        return sdk
    except ImportError:
        print_warning(
            "robot_interface SDK not found. Set UNITREE_SDK_PATH to the "
            "directory containing robot_interface.so to enable real-hardware "
            "motion. Running without SDK (motion commands are no-ops)."
        )
        return None


class RobotController:
    """Real-hardware controller for the Unitree Go1 quadruped."""

    def __init__(self):
        # Lazy ROS import so this module is importable without ROS installed.
        import rospy

        # Initialize the ROS node if it is not already running.
        if not rospy.get_node_uri():
            rospy.init_node('point_navigation', anonymous=True)

        # Initialize the Unitree SDK (UDP + command structures).
        self._sdk = _import_unitree_sdk()
        self._init_unitree_sdk()

        # Camera data structure (4 head cameras).
        self.num_cameras = Config.NUM_DIRECTIONS
        self.camera_data = {}
        for i in range(1, self.num_cameras + 1):
            self.camera_data[f'camera{i}'] = {
                'rgb_image': None,
                'depth_image': None,
                'camera_info': None,
                'camera_info_received': False,
                'fx': None, 'fy': None, 'cx': None, 'cy': None
            }

        # Directional image buffers (front / right / behind / left).
        self.direction_images = {
            'front': {'rgb': None, 'depth': None},
            'right': {'rgb': None, 'depth': None},
            'behind': {'rgb': None, 'depth': None},
            'left': {'rgb': None, 'depth': None}
        }

        self.current_direction = 'front'
        self.current_camera = 'camera1'
        self.running = True

        # Initialize the planner client (URL from Config.IPLANNER_URL).
        self.planner_client = IPlannerRemoteClient(Config.IPLANNER_URL)
        try:
            if not self.planner_client.reset():
                print_warning("Failed to reset planner client initially (server might be down). Will retry later.")
        except Exception as e:
            print_warning(f"Failed to reset planner client: {e}")

        # Set up ROS subscribers and publishers.
        self._setup_ros()

        # Navigation params.
        self.basic_turn_angle = 30
        self.planning_lock = threading.Lock()

        # History of saved iPlanner visualization images.
        self.iplanner_history = []

        print_info(f"RobotController Initialized with {self.num_cameras} cameras")

        # Background image-saver thread.
        self.saver_running = True
        self.saver_thread = threading.Thread(target=self._image_saver_loop)
        self.saver_thread.daemon = True
        self.saver_thread.start()

    def _image_saver_loop(self):
        """Background loop to save front camera images at ~10 FPS."""
        import cv2

        # Wait until Config.LOG_DIR is populated by main.py.
        attempts = 0
        base_dir = None
        while attempts < 10:
            if hasattr(Config, 'LOG_DIR') and Config.LOG_DIR:
                base_dir = Config.LOG_DIR
                break
            time.sleep(0.5)
            attempts += 1

        if not base_dir:
            # Fallback if config is never set (e.g. unit tests).
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", timestamp)
            base_dir = os.path.normpath(base_dir)

        save_dir = os.path.join(base_dir, "images", "front_view")
        os.makedirs(save_dir, exist_ok=True)
        print_info(f"Starting 10 FPS Image Saver. Saving to: {save_dir}")

        fps = 10.0
        interval = 1.0 / fps

        while self.saver_running:
            # self.saver_running is controlled by stop_saver(); loop until then.
            start_time = time.time()

            # Get the front camera image (camera1 is always front-facing).
            if 'camera1' in self.camera_data:
                img = self.camera_data['camera1']['rgb_image']

                if img is not None:
                    try:
                        # High-precision timestamp for unique filenames.
                        timestamp_str = datetime.now().strftime("%H%M%S_%f")
                        filename = f"front_{timestamp_str}.jpg"
                        filepath = os.path.join(save_dir, filename)
                        cv2.imwrite(filepath, img)
                    except Exception:
                        pass

            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

    def _init_unitree_sdk(self):
        """Initialize Unitree UDP and command structures."""
        if self._sdk is None:
            self.udp = None
            self.cmd = None
            self.state = None
            return
        try:
            HIGHLEVEL = 0xee
            self.udp = self._sdk.UDP(
                HIGHLEVEL,
                Config.UNITREE_LOCAL_PORT,
                Config.UNITREE_HOST,
                Config.UNITREE_REMOTE_PORT,
            )
            self.cmd = self._sdk.HighCmd()
            self.state = self._sdk.HighState()
            self.udp.InitCmdData(self.cmd)
        except Exception as e:
            print_error(f"Failed to initialize Unitree SDK: {e}")
            self.udp = None

    def _setup_ros(self):
        import rospy
        from sensor_msgs.msg import Image, CameraInfo
        from geometry_msgs.msg import Twist

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.subscribers = []

        for i in range(1, self.num_cameras + 1):
            # RGB
            rgb_sub = rospy.Subscriber(f'/camera{i}/color/image_raw', Image,
                                       self.rgb_callback, callback_args=f'camera{i}')
            # Depth (raw)
            depth_sub = rospy.Subscriber(f'/camera{i}/depth/image_raw', Image,
                                         self.depth_callback, callback_args=f'camera{i}')
            # Camera info
            camera_info_sub = rospy.Subscriber(f'/camera{i}/color/camera_info', CameraInfo,
                                               self.camera_info_callback, callback_args=f'camera{i}')

            self.subscribers.extend([rgb_sub, depth_sub, camera_info_sub])

    def rgb_callback(self, msg, camera_name):
        import cv2
        import numpy as np
        try:
            dtype = np.uint8
            n_channels = 3
            cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, n_channels)

            # Convert RGB to BGR if needed (assuming msg.encoding is rgb8).
            if 'rgb' in msg.encoding.lower():
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            self.camera_data[camera_name]['rgb_image'] = cv_image

            if camera_name == self.current_camera:
                self.rgb_image = cv_image
        except Exception:
            # Avoid spamming errors.
            pass

    def depth_callback(self, msg, camera_name):
        import numpy as np
        try:
            dtype = np.uint16
            cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
            self.camera_data[camera_name]['depth_image'] = cv_image

            if camera_name == self.current_camera:
                self.depth_image = cv_image
        except Exception:
            pass

    def camera_info_callback(self, msg, camera_name):
        data = self.camera_data[camera_name]
        if not data['camera_info_received']:
            data['camera_info'] = msg
            if msg.K and len(msg.K) >= 9:
                data['fx'] = msg.K[0]
                data['fy'] = msg.K[4]
                data['cx'] = msg.K[2]
                data['cy'] = msg.K[5]
                data['camera_info_received'] = True
                print_success(f"{camera_name} Intrinsics received")

    def get_depth_value(self, u, v, direction):
        """Get the depth value at specific pixel coordinates."""
        import numpy as np

        if self.direction_images[direction]['depth'] is None:
            print_warning(f"Depth image for {direction} is empty")
            return None

        try:
            depth_image = self.direction_images[direction]['depth']
            # Take a small window around the pixel to handle noise / missing values.
            k = 2
            # Ensure coordinates are within bounds.
            h, w = depth_image.shape
            u_min = max(0, u - k)
            u_max = min(w, u + k + 1)
            v_min = max(0, v - k)
            v_max = min(h, v + k + 1)

            depth_values = depth_image[v_min:v_max, u_min:u_max] / 1000.0  # Convert mm to meters.

            # Use the 30th percentile to avoid outliers (e.g. edges) but prefer
            # closer objects. Filter out zero values (invalid depth).
            valid_depths = depth_values[depth_values > 0]

            if len(valid_depths) > 0:
                depth = np.percentile(valid_depths, 30)
                if not np.isnan(depth):
                    return depth

            print_warning(f"Invalid depth at ({u}, {v})")
            return None
        except Exception as e:
            print_error(f"Error getting depth value: {e}")
            return None

    def capture_current_image(self) -> bool:
        """Capture the current camera image into the direction buffer."""
        curr_rgb = self.camera_data[self.current_camera]['rgb_image']
        curr_depth = self.camera_data[self.current_camera]['depth_image']

        if curr_rgb is not None and curr_depth is not None:
            self.direction_images[self.current_direction]['rgb'] = curr_rgb.copy()
            self.direction_images[self.current_direction]['depth'] = curr_depth.copy()
            print_action(f"Captured {self.current_direction} view using {self.current_camera}")
            return True
        print_warning(f"Failed to capture image for {self.current_direction}")
        return False

    def capture_all_directions(self, current_step: int) -> bool:
        """Switch the logical view across cameras to capture a full panorama.

        Captures the four directions front / right / behind / left from cameras
        1-4 at headings 0 / 90 / 180 / 270. The number of captured directions is
        driven by ``Config.NUM_DIRECTIONS`` (default 4).

        Each view is saved to ``Config.PANORAMA_DIR/step{current_step}/`` as
        ``view_0.png`` / ``view_90.png`` / ``view_180.png`` / ``view_270.png``
        and the in-memory frames remain available via ``direction_images``.

        Returns ``True`` if every requested direction was captured.
        """
        import cv2

        # Ordered direction / heading / camera mappings (front, right, behind, left).
        directions_all = ['front', 'right', 'behind', 'left']
        direction_mapping_all = {
            'front': 'view_0.png',
            'right': 'view_90.png',
            'behind': 'view_180.png',
            'left': 'view_270.png',
        }
        camera_mapping_all = {
            'front': 'camera1',
            'right': 'camera2',
            'behind': 'camera3',
            'left': 'camera4',
        }

        # Drive the captured count from configuration (default 4 views).
        n = max(1, min(Config.NUM_DIRECTIONS, len(directions_all)))
        directions = directions_all[:n]

        print_action(f"Step {current_step} - Capturing {n} directions ({', '.join(directions)})")

        step_panorama_dir = os.path.join(Config.PANORAMA_DIR, f"step{current_step}")
        os.makedirs(step_panorama_dir, exist_ok=True)

        for direction in directions:
            target_camera = camera_mapping_all[direction]
            if self.current_camera != target_camera:
                self.current_camera = target_camera
                self._sleep(0.5)  # Wait for stability.

            # Wait for data.
            self._sleep(0.2)

            self.current_direction = direction
            if not self.capture_current_image():
                print_warning(f"Retrying capture for {direction}...")
                self._sleep(1.0)
                if not self.capture_current_image():
                    return False

            # Save to disk.
            rgb_img = self.direction_images[direction]['rgb']
            if rgb_img is not None:
                # Resize if too large (match utils.py logic).
                h, w = rgb_img.shape[:2]
                max_size = 1024
                if h > max_size or w > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    rgb_img = cv2.resize(rgb_img, (new_w, new_h))

                filename = os.path.join(step_panorama_dir, direction_mapping_all[direction])
                cv2.imwrite(filename, rgb_img)

        # Reset to front.
        if self.current_camera != 'camera1':
            self.current_camera = 'camera1'
            self.current_direction = 'front'
            self._sleep(0.5)

        return True

    def get_front_image_jpeg(self) -> Optional[bytes]:
        """Encode the latest front (camera1) RGB frame as JPEG bytes.

        Used by the web MJPEG video feed. Returns ``None`` when no front frame
        is available yet or encoding fails.
        """
        import cv2

        img = self.camera_data.get('camera1', {}).get('rgb_image')
        if img is None:
            return None
        try:
            success, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if success:
                return buf.tobytes()
        except Exception as e:
            print_warning(f"Failed to encode front image to JPEG: {e}")
        return None

    def _sleep(self, seconds: float):
        """ROS-aware sleep that falls back to time.sleep without ROS."""
        try:
            import rospy
            rospy.sleep(seconds)
        except Exception:
            time.sleep(seconds)

    def rotate_left(self, angle):
        if not self.udp:
            return
        print_action(f"Rotating LEFT: {angle} degrees")
        self._perform_rotation(angle, direction=1)

    def rotate_right(self, angle):
        if not self.udp:
            return
        print_action(f"Rotating RIGHT: {angle} degrees")
        self._perform_rotation(angle, direction=-1)

    def _perform_rotation(self, angle, direction):
        """
        direction: 1 for left, -1 for right.
        """
        motiontime = 0
        # Correction factor for rotation overshoot.
        # Observed: 90 deg -> 120 deg (factor ~ 0.81).
        CORRECTION_FACTOR = 0.75
        corrected_angle = angle * CORRECTION_FACTOR

        # Calculate duration based on the corrected angle.
        duration = int(1000 * (corrected_angle / 90))
        yaw_speed = 0.87 * direction

        print_action(f"Corrected Rotation: {angle} -> {corrected_angle:.2f} deg (Duration: {duration} ms)")

        # Loop for duration.
        while motiontime < duration:
            time.sleep(0.002)
            motiontime += 1
            self.udp.Recv()
            self.udp.GetRecv(self.state)

            self.cmd.mode = 2
            self.cmd.gaitType = 1
            self.cmd.velocity = [0, 0]
            self.cmd.yawSpeed = yaw_speed
            self.cmd.footRaiseHeight = 0.1

            self.udp.SetSend(self.cmd)
            self.udp.Send()

        # Stop.
        self._send_stop_cmd()
        time.sleep(1.0)  # Wait for settle.
        print_success("Rotation complete")

    def stop_robot(self):
        print_action("Stopping robot (movement)...")
        # Do NOT stop saver_running here.

        if not self.udp:
            return
        for _ in range(10):
            self.udp.Recv()
            self.udp.GetRecv(self.state)
            self._send_stop_cmd()
            self.udp.Send()
            time.sleep(0.002)

    def stop_saver(self):
        """Explicitly stop the image-saver thread."""
        print_action("Stopping image saver thread...")
        self.saver_running = False

    def _send_stop_cmd(self):
        self.cmd.mode = 0
        self.cmd.gaitType = 0
        self.cmd.speedLevel = 0
        self.cmd.velocity = [0, 0]
        self.cmd.yawSpeed = 0
        self.udp.SetSend(self.cmd)

    def project_trajectory_to_image(self, trajectory_points, img_bgr, save_name="planned_path.png",
                                    target_pixel=None, target_3d=None, on_trajectory_image=None):
        """
        Project 3D trajectory points back to a 2D image and save it.

        :param trajectory_points: [N, 3] array (x, y, z).
        :param img_bgr: Original image.
        :param save_name: Filename to save.
        :param target_pixel: tuple (u, v) selected 2D pixel coordinates (optional).
        :param target_3d: tuple (x, y, z) calculated 3D coordinates (optional).
        :param on_trajectory_image: Optional callback receiving each recorded image path.
        """
        import cv2

        if trajectory_points is None or len(trajectory_points) == 0:
            return

        # Get intrinsics (camera1 is the main camera used for planning).
        cam_data = self.camera_data['camera1']
        fx, fy = cam_data['fx'], cam_data['fy']
        cx, cy = cam_data['cx'], cam_data['cy']

        if None in [fx, fy, cx, cy]:
            print_warning("Camera intrinsics missing, cannot draw trajectory")
            return

        vis_img = img_bgr.copy()
        h, w = vis_img.shape[:2]

        # --- 1. Draw the planned trajectory (green line) ---
        points_2d = []
        for pt in trajectory_points:
            # iPlanner output: pt[0]=Front(z), pt[1]=Left(x), y fixed.
            # Map to camera frame: Z_cam = Front, X_cam = -Left, Y_cam = Down (approx).
            # Standard ROS camera frame: Z forward, X right, Y down.
            # Robot frame: X forward, Y left, Z up.
            # So: Z_cam = X_robot, X_cam = -Y_robot, Y_cam = -Z_robot (or fixed height).

            sim_z = pt[0]         # Front -> Z_cam
            sim_x = -pt[1]        # Left -> -X_cam (camera X is right)

            # Get calibration params.
            cam_height = getattr(Config, 'CAMERA_HEIGHT', 0.3)
            roll_corr = getattr(Config, 'CAMERA_ROLL_CORRECTION', 0.0)

            # Apply correction:
            # Phenomenon: right-side (sim_x > 0) trajectory was too high (needs larger v -> larger sim_y);
            #             left-side (sim_x < 0) trajectory was too low (needs smaller v -> smaller sim_y).
            # Formula: sim_y = Height + sim_x * roll_corr.
            sim_y = cam_height + sim_x * roll_corr

            if sim_z <= 0.01:
                continue

            u = int((sim_x * fx) / sim_z + cx)
            v = int((sim_y * fy) / sim_z + cy)

            if 0 <= u < w and 0 <= v < h:
                points_2d.append((u, v))

        # Connect lines.
        if len(points_2d) > 1:
            for i in range(len(points_2d) - 1):
                cv2.line(vis_img, points_2d[i], points_2d[i + 1], (0, 255, 0), 2)
            # End point (red dot).
            cv2.circle(vis_img, points_2d[-1], 4, (0, 0, 255), -1)

        # --- 2. Draw the target pixel (red cross) ---
        if target_pixel is not None:
            tx, ty = target_pixel
            cv2.drawMarker(vis_img, (tx, ty), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # --- 3. Text ---
            text_lines = [f"Target Pixel: ({tx}, {ty})"]
            if target_3d:
                text_lines.append(f"Target 3D: ({target_3d[0]:.2f}, {target_3d[1]:.2f}, {target_3d[2]:.2f})m")

            font = cv2.FONT_HERSHEY_SIMPLEX
            base_y = ty + 25
            for line in text_lines:
                cv2.putText(vis_img, line, (tx + 10, base_y), font, 0.5, (0, 0, 0), 3)
                cv2.putText(vis_img, line, (tx + 10, base_y), font, 0.5, (0, 0, 255), 1)
                base_y += 20

        # Save.
        save_path = os.path.join(Config.IPLANNER_DIR, save_name)
        cv2.imwrite(save_path, vis_img)
        print_action(f"Saved iplanner visualization: {save_path}")

        # Append to history.
        self.iplanner_history.append(save_path)
        if on_trajectory_image is not None:
            on_trajectory_image(save_path)

    def execute_trajectory(self, trajectory_points, goal_local=None, on_trajectory_image=None):
        import numpy as np

        if trajectory_points is None or len(trajectory_points) < 2:
            print_warning("Trajectory too short to execute")
            return

        print_action(f"Executing trajectory with {len(trajectory_points)} points")

        target_speed = 0.3
        lookahead_dist = 0.4
        max_yaw_speed = 0.8
        goal_tolerance = 0.5
        replan_interval = 0.5

        path = np.array(trajectory_points)
        current_goal = path[-1].copy() if goal_local is None else np.array([goal_local[0], goal_local[1], 0.0])

        start_time = time.time()
        last_replan_time = start_time
        max_execution_time = 60.0
        current_idx = 0

        is_planning = False
        next_path = None

        def replan_worker(rgb, depth, goal):
            nonlocal next_path, is_planning
            try:
                traj, fear = self.planner_client.get_plan(rgb, depth, (goal[0], goal[1]))

                # Visualization and saving.
                try:
                    # Calculate the target pixel in the current image.
                    target_pixel_now = None
                    try:
                        cam_data = self.camera_data['camera1']
                        fx, fy = cam_data['fx'], cam_data['fy']
                        cx, cy = cam_data['cx'], cam_data['cy']

                        if None not in [fx, fy, cx, cy]:
                            # Robot frame (X=Forward, Y=Left) -> camera frame (Z=Forward, X=Right, Y=Down).
                            z_cam = goal[0]
                            x_cam = -goal[1]
                            y_cam = 0.3

                            if z_cam > 0.1:
                                u = int((x_cam * fx) / z_cam + cx)
                                v = int((y_cam * fy) / z_cam + cy)

                                h, w = rgb.shape[:2]
                                if 0 <= u < w and 0 <= v < h:
                                    target_pixel_now = (u, v)
                    except Exception:
                        pass

                    timestamp = datetime.now().strftime("%H%M%S_%f")
                    save_name = f"replan_{timestamp}.jpg"
                    self.project_trajectory_to_image(
                        traj, rgb, save_name=save_name, target_pixel=target_pixel_now,
                        target_3d=(goal[0], goal[1], 0.0), on_trajectory_image=on_trajectory_image)
                except Exception as vis_e:
                    print_warning(f"Visualization failed: {vis_e}")

                if fear is not None and fear <= 0.6 and traj is not None and len(traj) > 0:
                    with self.planning_lock:
                        next_path = traj
            except Exception as e:
                print_error(f"Replan failed: {e}")
            finally:
                is_planning = False

        while self.running:
            now = time.time()
            if now - start_time > max_execution_time:
                print_warning("Execution timeout")
                break

            # Check for a new path.
            with self.planning_lock:
                if next_path is not None:
                    path = np.array(next_path)
                    current_idx = 0
                    next_path = None

            # Trigger replan.
            if not is_planning and (now - last_replan_time > replan_interval):
                if self.capture_current_image():
                    curr_rgb = self.direction_images[self.current_direction]['rgb']
                    curr_depth = self.direction_images[self.current_direction]['depth']
                    if curr_rgb is not None:
                        is_planning = True
                        last_replan_time = now
                        t = threading.Thread(target=replan_worker, args=(curr_rgb.copy(), curr_depth.copy(), current_goal.copy()))
                        t.daemon = True
                        t.start()

            # Pure pursuit.
            found_target = False
            for i in range(current_idx, len(path)):
                if np.linalg.norm(path[i][:2]) > lookahead_dist:
                    current_idx = i
                    found_target = True
                    break

            if not found_target:
                current_idx = len(path) - 1

            target_point = path[current_idx]
            dx, dy = target_point[0], target_point[1]
            dist_to_target = math.sqrt(dx ** 2 + dy ** 2)
            dist_to_global_goal = np.linalg.norm(current_goal[:2])

            if dist_to_global_goal < goal_tolerance:
                print_success(f"Reached goal area (Dist: {dist_to_global_goal:.2f}m)")
                break

            # Control.
            alpha = math.atan2(dy, dx)
            cmd_yaw_speed = np.clip((2.0 * target_speed * math.sin(alpha)) / dist_to_target, -max_yaw_speed, max_yaw_speed)
            current_speed = max(0.1, target_speed * (1.0 - abs(alpha) / 3.14))

            # Send command.
            dt = 0.05
            loop_cycles = int(dt / 0.002)
            if self.udp:
                for _ in range(loop_cycles):
                    self.udp.Recv()
                    self.udp.GetRecv(self.state)
                    self.cmd.mode = 2
                    self.cmd.gaitType = 1
                    self.cmd.velocity = [current_speed, 0]
                    self.cmd.yawSpeed = cmd_yaw_speed
                    self.udp.SetSend(self.cmd)
                    self.udp.Send()
                    time.sleep(0.002)
            else:
                time.sleep(dt)

            # Dead-reckoning update.
            d_dist = current_speed * dt
            d_yaw = cmd_yaw_speed * dt
            cos_t, sin_t = math.cos(-d_yaw), math.sin(-d_yaw)

            # Update path points.
            new_path = []
            for p in path:
                px = p[0] * cos_t - p[1] * sin_t - d_dist
                py = p[0] * sin_t + p[1] * cos_t
                new_path.append([px, py, p[2]])
            path = np.array(new_path)

            # Update goal.
            gx = current_goal[0] * cos_t - current_goal[1] * sin_t - d_dist
            gy = current_goal[0] * sin_t + current_goal[1] * cos_t
            current_goal = np.array([gx, gy, current_goal[2]])

        self.stop_robot()
