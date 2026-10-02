"""LaViRA robot controller for the Unitree G1 humanoid.

Handles G1 locomotion control, 4x Orbbec Gemini 336L RGB-D capture, depth
processing, and trajectory execution.

Key differences from the Go1/Go2 adaptation:
- Uses ``LocoClient`` (g1_loco_client) instead of ``SportClient``.
- Camera height ~1.2 m (vs Go1 ~0.3 m).
- Uses ``ChannelFactoryInitialize`` instead of ``ChannelFactory.Initialize``.
- G1 exposes IMU state via a ``SportModeState_`` subscriber for yaw tracking.
- Humanoid-specific motion: StandUp, Squat, HighStand, etc.

Import safety
-------------
ROS (``rospy`` / ``scipy``), OpenCV (``cv2``), the native Unitree SDK
(``unitree_sdk2py``), and the Orbbec SDK (``pyorbbecsdk``) are imported lazily
inside ``__init__`` / methods so this module can be imported on a machine
without ROS, the SDKs, or OpenCV installed (``import robot.robot_controller``
succeeds with no hardware present).

Paths, network endpoints, and device serials
--------------------------------------------
The network interface comes from ``Config.NETWORK_INTERFACE``; the iPlanner
endpoint from ``Config.IPLANNER_URL``; the Orbbec camera serials from
``Config.serial_to_position()`` and ``Config.ORBBEC_REAR_SERIAL``. The module
ships no absolute paths and no embedded device literals.
"""
from __future__ import annotations

import glob
import math
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from utils import (
    print_action,
    print_error,
    print_info,
    print_robot,
    print_success,
    print_warning,
)
from config import Config
from robot.iplanner_client import IPlannerRemoteClient


def _import_ros():
    """Lazily import ROS / scipy. Returns ``(rospy, Odometry, R)`` or Nones."""
    try:
        import rospy
        from nav_msgs.msg import Odometry
        from scipy.spatial.transform import Rotation as R

        return rospy, Odometry, R
    except ImportError:
        print_warning("rospy or scipy not found. Odometry disabled.")
        return None, None, None


def _import_unitree_sdk():
    """Lazily import the Unitree G1 SDK. Returns a symbol dict or ``None``."""
    try:
        from unitree_sdk2py.core.channel import (
            ChannelSubscriber,
            ChannelFactoryInitialize,
        )
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__SportModeState_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        return {
            "ChannelSubscriber": ChannelSubscriber,
            "ChannelFactoryInitialize": ChannelFactoryInitialize,
            "SportModeState_": SportModeState_,
            "LocoClient": LocoClient,
        }
    except ImportError:
        print_warning(
            "unitree_sdk2py not found. Running in mock mode. "
            "Install with: pip3 install unitree_sdk2py"
        )
        return None


def _import_orbbec():
    """Lazily import the Orbbec SDK module. Returns the module or ``None``."""
    try:
        import pyorbbecsdk

        return pyorbbecsdk
    except ImportError:
        print_warning("pyorbbecsdk not found. Camera functions disabled.")
        return None


class RobotController:
    """Main controller for the Unitree G1 humanoid with 4x Orbbec Gemini 336L.

    Provides:
    - G1 locomotion control (walk, rotate, stop).
    - Orbbec Gemini 336L RGB-D capture (front, left, right, rear).
    - Simultaneous 4-way observation.
    - Trajectory following with pure pursuit assisted by ROS Odometry.
    - Depth-based 3D point unprojection.
    """

    def __init__(self):
        self.running = True

        # Serial -> position map (front / left / right), driven by Config.
        # The rear camera is identified separately by USB 2.0 connection /
        # dev path (see Config.ORBBEC_REAR_SERIAL / ORBBEC_REAR_DEV_FALLBACK).
        self.SERIAL_TO_POSITION = Config.serial_to_position()

        # Lazily-resolved SDK handles (populated in the init helpers below).
        self._ros = None  # rospy module
        self._Odometry = None
        self._R = None  # scipy Rotation
        self._sdk = None  # Unitree SDK symbol dict
        self._ob = None  # pyorbbecsdk module

        # =====================================================================
        # Initialize Unitree G1 SDK
        # =====================================================================
        self._init_unitree_g1()

        # =====================================================================
        # Initialize ROS Odometry
        # =====================================================================
        self._init_ros_odometry()

        # =====================================================================
        # Initialize Orbbec Cameras
        # =====================================================================
        self._init_orbbec_cameras()

        # =====================================================================
        # Camera Data Storage
        # =====================================================================
        self.rgb_image = None
        self.depth_image = None

        # Image Saver Setup
        self.save_images = True
        self.front_cam_dir = os.path.join(Config.IMG_DIR, "front_view")
        os.makedirs(self.front_cam_dir, exist_ok=True)
        self.frame_count = 0
        self.save_rgb_interval = 1.0 / Config.SAVE_RGB_FPS
        self.last_save_time = 0.0

        # Camera Intrinsics (Front Camera).
        # Default values in case capture has not happened yet.
        self.fx = 360.0  # Approx
        self.fy = 360.0
        self.cx = 320.0
        self.cy = 240.0

        # Directional Images
        self.direction_images = {
            "front": {"rgb": None, "depth": None},
            "right": {"rgb": None, "depth": None},
            "behind": {"rgb": None, "depth": None},
            "left": {"rgb": None, "depth": None},
        }

        self.current_direction = "front"
        self.current_camera = "camera1"

        # =====================================================================
        # IMU / State Tracking
        # =====================================================================
        self.current_yaw = 0.0  # radians, from IMU
        self.last_imu_time = 0.0  # timestamp of last IMU update
        self.imu_lock = threading.Lock()
        self._start_state_subscriber()

        # =====================================================================
        # iPlanner Client
        # =====================================================================
        self.planner_client = IPlannerRemoteClient(Config.IPLANNER_URL)

        # Try to reset planner with intrinsics if available.
        if self.fx:
            intrinsics = [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
            try:
                self.planner_client.reset(intrinsic=intrinsics)
            except Exception as e:
                print_warning(f"Failed to reset planner client: {e}")

        # =====================================================================
        # Navigation State
        # =====================================================================
        self.planning_lock = threading.Lock()
        self.iplanner_history = []

        # =====================================================================
        # Control Thread (continuous velocity command sending)
        # =====================================================================
        self.cmd_lock = threading.Lock()
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vyaw = 0.0
        self.control_thread = threading.Thread(
            target=self._control_loop, daemon=True
        )
        self.control_thread.start()

        print_success("RobotController Initialized (G1 Humanoid + 4x Orbbec)")

    # =========================================================================
    # Initialization Methods
    # =========================================================================

    def _init_unitree_g1(self):
        """Initialize Unitree G1 SDK2 LocoClient."""
        self._sdk = _import_unitree_sdk()
        if self._sdk is None:
            print_warning("Unitree SDK not available, running in mock mode")
            self.loco_client = None
            return

        try:
            self._sdk["ChannelFactoryInitialize"](0, Config.NETWORK_INTERFACE)

            self.loco_client = self._sdk["LocoClient"]()
            self.loco_client.SetTimeout(10.0)
            self.loco_client.Init()

            print_success(
                f"Unitree G1 SDK2 Initialized "
                f"(interface: {Config.NETWORK_INTERFACE})"
            )
        except Exception as e:
            print_error(f"Failed to initialize Unitree G1 SDK2: {e}")
            self.loco_client = None

    def _init_ros_odometry(self):
        """Initialize the ROS node and subscribe to ``/Odometry``."""
        self.current_pose = None
        self.odom_lock = threading.Lock()

        self._ros, self._Odometry, self._R = _import_ros()
        if self._ros is None:
            return

        if not Config.USE_LIDAR:
            print_warning(
                "LiDAR Odometry disabled by config. Using Dead Reckoning."
            )
            return

        try:
            rospy = self._ros
            # Check if node is already initialized.
            if rospy.get_node_uri() is None:
                rospy.init_node(
                    "lavira_controller", anonymous=True, disable_signals=True
                )

            self.odom_sub = rospy.Subscriber(
                "/Odometry", self._Odometry, self._odometry_callback, queue_size=1
            )
            print_success("ROS Odometry subscriber initialized")
        except Exception as e:
            print_error(f"Failed to initialize ROS Odometry: {e}")

    def _odometry_callback(self, msg):
        """Callback for the ``/Odometry`` topic."""
        try:
            # Extract position.
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z

            # CRITICAL FAILURE CHECK: If SLAM has diverged (e.g. huge Z),
            # disable Odometry.
            if abs(z) > 5.0:  # Robot cannot fly 5m high
                if not hasattr(self, "odom_disabled"):
                    self.odom_disabled = True
                    print_error(
                        f"SLAM Divergence Detected (Z={z:.1f}m)! Disabling ROS "
                        "Odometry permanently for this session."
                    )
                return

            if hasattr(self, "odom_disabled") and self.odom_disabled:
                return

            # SANITY CHECK: Reset origin if values are too large.
            if abs(x) > 1000.0 or abs(y) > 1000.0:
                if not hasattr(self, "odom_offset"):
                    self.odom_offset = np.array([x, y])
                    print_warning(
                        f"Large Odometry detected. Setting offset: "
                        f"{self.odom_offset}"
                    )

                x -= self.odom_offset[0]
                y -= self.odom_offset[1]
            else:
                if hasattr(self, "odom_offset"):
                    x -= self.odom_offset[0]
                    y -= self.odom_offset[1]

            # Extract orientation (quaternion).
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w

            # Convert to Euler (yaw).
            rot = self._R.from_quat([qx, qy, qz, qw])
            rpy = rot.as_euler("xyz", degrees=False)
            yaw = rpy[2]

            with self.odom_lock:
                self.current_pose = np.array([x, y, yaw])

        except Exception:
            pass

    def _init_orbbec_cameras(self):
        """Initialize 4x Orbbec Gemini 336L cameras (device discovery only)."""
        import cv2  # Lazy import: keep this module importable without OpenCV.

        self.camera_devices = {}  # Store device objects/indices for serial access
        self.rear_dev_path = None
        self.front_pipeline = None
        self.front_camera_thread = None
        self.front_camera_lock = threading.Lock()

        self._ob = _import_orbbec()
        if self._ob is None:
            return

        # 1. Identify Rear Camera (USB 2.0) via V4L2/Serial.
        # We do this BEFORE SDK initialization.
        self.rear_dev_path = self._find_rear_camera_v4l2()

        if self.rear_dev_path:
            print_info(f"Rear Camera found at {self.rear_dev_path} (V4L2)")
            # Pre-init/Warmup check.
            self._capture_single_camera("behind")
        else:
            print_warning("Rear Camera not found via V4L2. Will check SDK.")

        # 2. Initialize SDK Context.
        self.ctx = self._ob.Context()
        self.ctx.set_logger_level(self._ob.OBLogLevel.WARNING)

        # 3. Query Devices.
        try:
            device_list = self.ctx.query_devices()
            device_count = device_list.get_count()
            print_info(f"Found {device_count} Orbbec devices via SDK.")

            # Store device list for later access (cannot store Device objects
            # directly for long periods easily). We will re-query or store
            # indices. Storing indices is safer if the list does not change,
            # but storing a serial -> index mapping is better.
            self.sdk_devices = []
            for i in range(device_count):
                device = device_list.get_device_by_index(i)
                serial = device.get_device_info().get_serial_number()
                self.sdk_devices.append({"index": i, "serial": serial})

                # Map to position.
                pos = self.SERIAL_TO_POSITION.get(serial)
                if pos:
                    print_success(f"  Mapped SDK Device {i} (SN:{serial}) -> {pos}")
                else:
                    # Check if it might be Rear (USB 2.0).
                    ctype = str(
                        device.get_device_info().get_connection_type()
                    ).upper()
                    if "2.0" in ctype or "USB2" in ctype:
                        print_info(
                            f"  SDK Device {i} (SN:{serial}) appears to be Rear "
                            "(USB 2.0)"
                        )
                        # If we did not find V4L2, maybe mark this?
                        # But SDK streaming for Rear is what we want to AVOID.
                    else:
                        print_warning(f"  Unmapped Device {i} (SN:{serial})")

            # 4. Start Front Camera Stream (Persistent).
            self._start_front_camera_stream()

        except Exception as e:
            print_error(f"Failed to query SDK devices: {e}")

    def _start_front_camera_stream(self):
        """Start a persistent stream for the Front Camera."""
        OBSensorType = self._ob.OBSensorType
        OBFormat = self._ob.OBFormat

        target_serial = None
        for s, p in self.SERIAL_TO_POSITION.items():
            if p == "front":
                target_serial = s
                break

        if not target_serial:
            return

        try:
            device_list = self.ctx.query_devices()
            device_count = device_list.get_count()
            device = None

            for i in range(device_count):
                d = device_list.get_device_by_index(i)
                try:
                    sn = d.get_device_info().get_serial_number()
                    if sn == target_serial:
                        device = d
                        break
                except Exception:
                    continue

            if device:
                self.front_pipeline = self._ob.Pipeline(device)
                config = self._ob.Config()

                # RGB
                profiles = self.front_pipeline.get_stream_profile_list(
                    OBSensorType.COLOR_SENSOR
                )
                try:
                    rgb_profile = profiles.get_video_stream_profile(
                        640, 480, OBFormat.RGB, 30
                    )
                except Exception:
                    rgb_profile = profiles.get_profile_by_index(0)
                config.enable_stream(rgb_profile)

                # Depth
                try:
                    d_profiles = self.front_pipeline.get_stream_profile_list(
                        OBSensorType.DEPTH_SENSOR
                    )
                    try:
                        d_profile = d_profiles.get_video_stream_profile(
                            640, 480, OBFormat.Y16, 30
                        )
                    except Exception:
                        d_profile = d_profiles.get_profile_by_index(0)
                    config.enable_stream(d_profile)
                except Exception:
                    pass

                self.front_pipeline.start(config)

                # Start update thread.
                self.front_camera_thread = threading.Thread(
                    target=self._update_front_camera_loop, daemon=True
                )
                self.front_camera_thread.start()
                print_success("Front Camera Persistent Stream Started")
        except Exception as e:
            print_error(f"Failed to start Front Camera stream: {e}")

    def _update_front_camera_loop(self):
        """Background thread to update front camera frames."""
        import cv2

        while self.running and self.front_pipeline:
            try:
                frames = self.front_pipeline.wait_for_frames(100)
                if frames:
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if color_frame:
                        data = color_frame.get_data()
                        w, h = color_frame.get_width(), color_frame.get_height()
                        rgb = np.frombuffer(data, dtype=np.uint8).reshape(
                            (h, w, 3)
                        )
                        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                        with self.front_camera_lock:
                            self.rgb_image = rgb_bgr

                            # Get intrinsics once.
                            if self.fx == 360.0:  # If default
                                try:
                                    param = self.front_pipeline.get_camera_param()
                                    self.fx = param.rgb_intrinsic.fx
                                    self.fy = param.rgb_intrinsic.fy
                                    self.cx = param.rgb_intrinsic.cx
                                    self.cy = param.rgb_intrinsic.cy
                                except Exception:
                                    pass

                        # Save Frame (save RGB images at the configured fps).
                        if self.save_images:
                            current_time = time.time()
                            if (
                                current_time - self.last_save_time
                                >= self.save_rgb_interval
                            ):
                                timestamp = datetime.now().strftime("%H%M%S_%f")
                                filename = os.path.join(
                                    self.front_cam_dir,
                                    f"frame_{self.frame_count:06d}_{timestamp}.jpg",
                                )
                                # Save BGR image (cv2.imwrite expects BGR).
                                cv2.imwrite(filename, rgb_bgr)
                                self.frame_count += 1
                                self.last_save_time = current_time

                    if depth_frame:
                        data = depth_frame.get_data()
                        w, h = depth_frame.get_width(), depth_frame.get_height()
                        depth_img = np.frombuffer(data, dtype=np.uint16).reshape(
                            (h, w)
                        )

                        with self.front_camera_lock:
                            self.depth_image = depth_img

            except Exception:
                pass
            time.sleep(0.005)

    def _find_rear_camera_v4l2(self) -> Optional[str]:
        """Find the Rear camera ``/dev/videoX`` path by serial or fallback."""
        rear_serials = [
            s for s in [Config.ORBBEC_REAR_SERIAL] if s
        ]  # Known rear serial(s)

        # Helper to find a device by serial.
        def get_dev_by_serial(serials):
            video_devs = sorted(glob.glob("/dev/video*"))
            for dev in video_devs:
                try:
                    cmd = ["udevadm", "info", "--query=all", f"--name={dev}"]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    for s in serials:
                        if f"ID_SERIAL_SHORT={s}" in res.stdout:
                            # Check MJPG support.
                            cmd_fmt = ["v4l2-ctl", "-d", dev, "--list-formats"]
                            res_fmt = subprocess.run(
                                cmd_fmt, capture_output=True, text=True
                            )
                            if "MJPG" in res_fmt.stdout:
                                return dev
                except Exception:
                    continue
            return None

        if rear_serials:
            dev = get_dev_by_serial(rear_serials)
            if dev:
                return dev

        # Fallback.
        fallback = Config.ORBBEC_REAR_DEV_FALLBACK
        if fallback and os.path.exists(fallback):
            return fallback
        return None

    def _capture_single_camera(
        self, position: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture a single frame from a specific camera (serial execution).

        Includes a warmup to fix dark images.
        """
        import cv2

        rgb_img = None
        depth_img = None

        # --- Case 1: Rear Camera (V4L2) ---
        if position == "behind":
            if not self.rear_dev_path:
                print_warning("Rear camera path not set.")
                return None, None

            try:
                cap = cv2.VideoCapture(self.rear_dev_path)
                if cap.isOpened():
                    cap.set(
                        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
                    )
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    # Warmup (Critical for Auto-Exposure).
                    # For V4L2, 30-60 frames is good.
                    for _ in range(30):
                        cap.read()

                    ret, frame = cap.read()
                    if ret:
                        rgb_img = frame  # BGR
                    cap.release()
                else:
                    print_error(f"Failed to open {self.rear_dev_path}")
            except Exception as e:
                print_error(f"Rear capture error: {e}")

            return rgb_img, None  # No depth for Rear V4L2

        # --- Case 2: Front/Left/Right (SDK) ---
        OBSensorType = self._ob.OBSensorType
        OBFormat = self._ob.OBFormat

        # Find device index.
        target_serial = None
        for s, p in self.SERIAL_TO_POSITION.items():
            if p == position:
                target_serial = s
                break

        if not target_serial:
            print_error(f"No serial found for position {position}")
            return None, None

        device_index = -1
        for d in self.sdk_devices:
            if d["serial"] == target_serial:
                device_index = d["index"]
                break

        if device_index == -1:
            print_error(
                f"Device for {position} (SN:{target_serial}) not found in SDK "
                "list."
            )
            return None, None

        # Create Pipeline & Capture.
        try:
            # FIX: Do NOT use device_index directly from query_devices() as the
            # order might change or the list might be empty. Instead, find the
            # device by serial again to be safe.
            device_list = self.ctx.query_devices()
            device_count = device_list.get_count()
            device = None

            for i in range(device_count):
                d = device_list.get_device_by_index(i)
                try:
                    sn = d.get_device_info().get_serial_number()
                    if sn == target_serial:
                        device = d
                        break
                except Exception:
                    continue

            if device is None:
                print_error(
                    f"Device with serial {target_serial} not found during "
                    "capture."
                )
                return None, None

            pipeline = self._ob.Pipeline(device)
            config = self._ob.Config()

            # RGB
            profiles = pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            try:
                rgb_profile = profiles.get_video_stream_profile(
                    640, 480, OBFormat.RGB, 30
                )
            except Exception:
                rgb_profile = profiles.get_profile_by_index(0)
            config.enable_stream(rgb_profile)

            # Depth (only for Front usually, but we can try for others if
            # needed). Front needs depth for navigation.
            enable_depth = position == "front"
            if enable_depth:
                try:
                    d_profiles = pipeline.get_stream_profile_list(
                        OBSensorType.DEPTH_SENSOR
                    )
                    try:
                        d_profile = d_profiles.get_video_stream_profile(
                            640, 480, OBFormat.Y16, 30
                        )
                    except Exception:
                        d_profile = d_profiles.get_profile_by_index(0)
                    config.enable_stream(d_profile)
                except Exception:
                    pass

            pipeline.start(config)

            # Warmup (Critical for Auto-Exposure).
            # 60 frames ~ 2 seconds.
            for _ in range(60):
                pipeline.wait_for_frames(100)

            # Capture.
            frames = pipeline.wait_for_frames(2000)
            if frames:
                color_frame = frames.get_color_frame()
                if color_frame:
                    data = color_frame.get_data()
                    w, h = color_frame.get_width(), color_frame.get_height()
                    rgb = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                    rgb_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        data = depth_frame.get_data()
                        w, h = (
                            depth_frame.get_width(),
                            depth_frame.get_height(),
                        )
                        depth_img = np.frombuffer(
                            data, dtype=np.uint16
                        ).reshape((h, w))

                # Get intrinsics if Front.
                if position == "front" and rgb_img is not None:
                    try:
                        param = pipeline.get_camera_param()
                        self.fx = param.rgb_intrinsic.fx
                        self.fy = param.rgb_intrinsic.fy
                        self.cx = param.rgb_intrinsic.cx
                        self.cy = param.rgb_intrinsic.cy
                    except Exception:
                        pass

            pipeline.stop()

        except Exception as e:
            print_error(f"Error capturing {position}: {e}")

        return rgb_img, depth_img

    def _start_state_subscriber(self):
        """Subscribe to G1 robot state for IMU yaw tracking."""
        if self._sdk is None:
            return

        try:
            self.state_sub = self._sdk["ChannelSubscriber"](
                "rt/sportmodestate", self._sdk["SportModeState_"]
            )
            self.state_sub.Init(self._state_callback, 10)
            print_info("G1 state subscriber initialized (IMU yaw tracking)")
        except Exception as e:
            print_warning(f"Failed to initialize state subscriber: {e}")
            self.state_sub = None

    def _state_callback(self, msg):
        """Callback for robot state updates (IMU data)."""
        try:
            with self.imu_lock:
                # SportModeState_ contains imu_state with a quaternion or rpy.
                # yaw is the rotation around the vertical axis.
                if hasattr(msg, "imu_state") and hasattr(
                    msg.imu_state, "rpy"
                ):
                    self.current_yaw = msg.imu_state.rpy[2]  # yaw in radians
                    self.last_imu_time = time.time()
        except Exception:
            pass

    # =========================================================================
    # Control Loop
    # =========================================================================

    def _control_loop(self):
        """Background thread to send velocity commands continuously to G1."""
        while self.running:
            if self.loco_client:
                with self.cmd_lock:
                    vx = self.target_vx
                    vy = self.target_vy
                    vyaw = self.target_vyaw

                try:
                    # G1 LocoClient.Move(vx, vy, vyaw).
                    self.loco_client.Move(vx, vy, vyaw)
                except Exception:
                    # Silently handle to avoid log spam.
                    pass
            time.sleep(0.02)  # 50Hz control loop

    def _set_velocity(self, vx, vy, vyaw):
        """Thread-safe velocity command update."""
        with self.cmd_lock:
            self.target_vx = vx
            self.target_vy = vy
            self.target_vyaw = vyaw

    # =========================================================================
    # Camera Methods
    # =========================================================================

    def update_camera_data(self) -> bool:
        """Fetch the latest RGB-D frames from the Front Orbbec camera."""
        # Check if the persistent stream is active.
        if self.front_pipeline and self.front_camera_thread:
            with self.front_camera_lock:
                if self.rgb_image is not None:
                    # Already updated by the background thread.

                    # Store for compatibility.
                    self.camera_data = {
                        "camera1": {
                            "rgb_image": self.rgb_image,
                            "depth_image": self.depth_image,
                            "fx": self.fx,
                            "fy": self.fy,
                            "cx": self.cx,
                            "cy": self.cy,
                        }
                    }

                    # Update planner client intrinsics if needed.
                    if self.fx and not self.planner_client.initialized:
                        intrinsics = [
                            [self.fx, 0.0, self.cx],
                            [0.0, self.fy, self.cy],
                            [0.0, 0.0, 1.0],
                        ]
                        try:
                            self.planner_client.reset(intrinsic=intrinsics)
                        except Exception:
                            pass
                    return True
                else:
                    return False

        # Fallback to slow serial capture if the stream failed.
        rgb, depth = self._capture_single_camera("front")

        if rgb is not None:
            self.rgb_image = rgb
            self.depth_image = depth  # Might be None if depth failed

            # Store for compatibility.
            self.camera_data = {
                "camera1": {
                    "rgb_image": self.rgb_image,
                    "depth_image": self.depth_image,
                    "fx": self.fx,
                    "fy": self.fy,
                    "cx": self.cx,
                    "cy": self.cy,
                }
            }

            # Update planner client intrinsics if needed.
            if self.fx and not self.planner_client.initialized:
                intrinsics = [
                    [self.fx, 0.0, self.cx],
                    [0.0, self.fy, self.cy],
                    [0.0, 0.0, 1.0],
                ]
                try:
                    self.planner_client.reset(intrinsic=intrinsics)
                except Exception:
                    pass
            return True
        else:
            return False

    def get_depth_value(self, u, v, direction):
        """Get the depth value at specific pixel coordinates.

        Uses a small window around the target pixel and takes the 30th
        percentile of valid depths for robustness against noise.

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.
            direction: Direction key ('front', 'right', 'behind', 'left').

        Returns:
            Depth in metres, or ``None`` if unavailable.
        """
        if self.direction_images[direction]["depth"] is None:
            return None

        try:
            depth_image = self.direction_images[direction]["depth"]
            h, w = depth_image.shape

            # Boundary check.
            if u < 0 or u >= w or v < 0 or v >= h:
                return None

            # Take a small window for robustness.
            k = 3
            u_min = max(0, u - k)
            u_max = min(w, u + k + 1)
            v_min = max(0, v - k)
            v_max = min(h, v + k + 1)

            # Orbbec depth is in mm (uint16); convert to metres.
            depth_values = (
                depth_image[v_min:v_max, u_min:u_max].astype(float) / 1000.0
            )

            valid_depths = depth_values[depth_values > 0.1]  # Filter out invalid
            if len(valid_depths) > 0:
                depth = np.percentile(valid_depths, 30)
                return depth
            return None
        except Exception:
            return None

    def capture_current_image(self) -> bool:
        """Capture the current front camera image to the direction buffer."""
        if self.update_camera_data():
            # Update the buffer for the CURRENT direction the robot is facing
            # (since we rotated, the Front camera now captures this direction).
            self.direction_images[self.current_direction][
                "rgb"
            ] = self.rgb_image.copy()
            self.direction_images[self.current_direction][
                "depth"
            ] = self.depth_image.copy()
            return True
        return False

    def capture_all_directions(self, current_step: int) -> bool:
        """Capture a simultaneous 360-degree panorama from 4 Orbbec cameras.

        (Serial implementation to save bandwidth.)
        """
        import cv2

        print_action(
            f"Step {current_step} - Capturing 360-degree panorama (Serial Mode)"
        )

        directions = ["front", "right", "behind", "left"]
        direction_mapping = {
            "front": "view_0.png",
            "right": "view_90.png",
            "behind": "view_180.png",
            "left": "view_270.png",
        }

        step_panorama_dir = os.path.join(
            Config.PANORAMA_DIR, f"step{current_step}"
        )
        os.makedirs(step_panorama_dir, exist_ok=True)

        success_count = 0

        for direction in directions:
            print_info(f"  Capturing {direction}...")

            # Special case for Front: use persistent stream if available.
            if direction == "front" and self.front_pipeline:
                if self.update_camera_data():
                    rgb = self.rgb_image
                    depth = self.depth_image
                else:
                    rgb, depth = None, None
            else:
                rgb, depth = self._capture_single_camera(direction)

            if rgb is not None:
                self.direction_images[direction]["rgb"] = rgb
                if depth is not None:
                    self.direction_images[direction]["depth"] = depth

                # Save.
                filename = os.path.join(
                    step_panorama_dir, direction_mapping[direction]
                )
                cv2.imwrite(filename, rgb)
                success_count += 1
            else:
                print_warning(f"  Failed to capture {direction}")

            # Fallback for missing cameras.
            img_bgr = self.direction_images[direction]["rgb"]
            if img_bgr is None:
                print_warning(
                    f"Missing image for {direction}, creating placeholder."
                )
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    f"{direction} view unavailable",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                self.direction_images[direction]["rgb"] = placeholder

                filename = os.path.join(
                    step_panorama_dir, direction_mapping[direction]
                )
                cv2.imwrite(filename, placeholder)

        print_success(
            f"Step {current_step} - Panorama capture complete "
            f"({success_count}/4 cameras)"
        )
        return True

    def get_front_image_jpeg(self) -> Optional[bytes]:
        """Encode the latest front camera RGB frame as JPEG bytes.

        Used by the web MJPEG video feed. Returns ``None`` when no front frame
        is available yet or encoding fails.
        """
        import cv2

        with self.front_camera_lock:
            img = self.rgb_image
        if img is None:
            return None
        try:
            success, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            )
            if success:
                return buf.tobytes()
        except Exception as e:
            print_warning(f"Failed to encode front image to JPEG: {e}")
        return None

    # =========================================================================
    # Motion Control Methods
    # =========================================================================

    def rotate_left(self, angle_deg):
        """Rotate G1 in place to the left (counter-clockwise)."""
        print_robot(f"Rotating LEFT: {angle_deg} degrees")
        self._perform_rotation(angle_deg, direction_sign=1.0)

    def rotate_right(self, angle_deg):
        """Rotate G1 in place to the right (clockwise)."""
        print_robot(f"Rotating RIGHT: {angle_deg} degrees")
        self._perform_rotation(angle_deg, direction_sign=-1.0)

    def _perform_rotation(self, angle_deg, direction_sign):
        """Perform an in-place rotation using IMU feedback or open-loop timing.

        For the G1 humanoid we use a conservative rotation speed to maintain
        balance. If the IMU is available, we use closed-loop yaw control for
        accuracy.

        Args:
            angle_deg: Rotation angle in degrees.
            direction_sign: +1.0 for left (CCW), -1.0 for right (CW).
        """
        target_angle_rad = math.radians(angle_deg)
        yaw_speed = Config.ROTATION_SPEED * direction_sign

        use_imu = False
        if (
            Config.USE_IMU_FOR_ROTATION
            and hasattr(self, "state_sub")
            and self.state_sub
        ):
            # Check if the IMU is active (data received recently).
            if time.time() - self.last_imu_time < 1.0:
                use_imu = True
            else:
                print_warning(
                    "IMU data timeout/inactive. Falling back to open-loop "
                    "rotation."
                )

        if use_imu:
            # Closed-loop rotation using IMU yaw feedback.
            self._rotate_with_imu(target_angle_rad, yaw_speed, direction_sign)
        else:
            # Open-loop rotation based on timing.
            self._rotate_open_loop(target_angle_rad, yaw_speed)

    def _rotate_with_imu(self, target_angle_rad, yaw_speed, direction_sign):
        """IMU-based closed-loop rotation for precise angle control."""
        with self.imu_lock:
            start_yaw = self.current_yaw

        accumulated = 0.0
        prev_yaw = start_yaw
        timeout = (
            target_angle_rad / abs(yaw_speed) * 2.0 + 5.0
        )  # generous timeout
        start_time = time.time()

        while abs(accumulated) < target_angle_rad:
            if time.time() - start_time > timeout:
                print_warning("Rotation timeout, stopping")
                break

            self._set_velocity(0.0, 0.0, yaw_speed)
            time.sleep(0.02)

            with self.imu_lock:
                current_yaw = self.current_yaw

            # Calculate delta yaw (handle wraparound).
            delta = current_yaw - prev_yaw
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi

            accumulated += delta * direction_sign
            prev_yaw = current_yaw

        self.stop_robot()
        time.sleep(0.5)  # Settle

    def _rotate_open_loop(self, target_angle_rad, yaw_speed):
        """Open-loop rotation based on timing."""
        # Scale duration because the G1 actual rotation speed is often slower
        # than commanded.
        # Observation: 3x 90deg commands resulted in 180deg actual rotation ->
        # 1.5x scale needed. Adjusted to 1.4 to fix slight over-rotation.
        duration_scale = 1.4
        duration = (target_angle_rad / abs(yaw_speed)) * duration_scale

        start_time = time.time()
        while time.time() - start_time < duration:
            self._set_velocity(0.0, 0.0, yaw_speed)
            time.sleep(0.02)

        self.stop_robot()
        time.sleep(0.5)  # Settle

    def stop_robot(self):
        """Stop all robot motion."""
        self._set_velocity(0.0, 0.0, 0.0)
        if self.loco_client:
            try:
                self.loco_client.StopMove()
            except Exception:
                pass
        time.sleep(0.2)

    def stand_up(self):
        """Command G1 to stand up from the squat position."""
        if self.loco_client:
            try:
                self.loco_client.Squat2StandUp()
                print_robot("Standing up...")
                time.sleep(3.0)
            except Exception as e:
                print_error(f"StandUp failed: {e}")

    def high_stand(self):
        """Command G1 to the high stand posture."""
        if self.loco_client:
            try:
                self.loco_client.HighStand()
                print_robot("High stand posture")
                time.sleep(1.0)
            except Exception as e:
                print_error(f"HighStand failed: {e}")

    # =========================================================================
    # Trajectory Execution
    # =========================================================================

    def execute_trajectory(self, trajectory_points, goal_local=None, on_trajectory_image=None):
        """Execute a planned trajectory using pure pursuit with continuous
        re-planning.

        Assisted by ROS Odometry if available.

        The trajectory is in the robot's local frame:
        - X axis: forward
        - Y axis: left
        - Z axis: up

        Args:
            trajectory_points: Initial list of ``[x, y, z]`` waypoints in the
                robot frame.
            goal_local: Optional ``(x, y)`` goal override for the termination
                check.
            on_trajectory_image: Optional callback receiving each recorded image path.
        """
        if trajectory_points is None or len(trajectory_points) < 2:
            print_warning("Trajectory too short to execute")
            return

        print_robot(
            f"Executing trajectory with continuous re-planning "
            f"(Interval: {Config.REPLAN_INTERVAL}s)"
        )

        # Clear iplanner_history before starting continuous execution to avoid
        # mixing with the previous step's static plan. BUT we want to keep them
        # for "Visual History". Let's just ensure we append the replan images.

        # Pure pursuit parameters (tuned for the G1 humanoid).
        target_speed = Config.DEFAULT_WALK_SPEED
        lookahead_dist = 0.5  # Slightly larger for humanoid stability
        max_yaw_speed = Config.MAX_YAW_SPEED
        goal_tolerance = Config.GOAL_TOLERANCE

        path = np.array(trajectory_points)
        current_goal = (
            path[-1].copy()
            if goal_local is None
            else np.array([goal_local[0], goal_local[1], 0.0])
        )

        # Initialize Odometry-based tracking if available.
        use_odom = False
        global_goal = None

        # Check if Odometry is valid (not disabled due to divergence).
        odom_valid = self._ros is not None and self.current_pose is not None
        if hasattr(self, "odom_disabled") and self.odom_disabled:
            odom_valid = False

        if odom_valid:
            use_odom = True
            with self.odom_lock:
                start_pose = self.current_pose.copy()  # [x, y, yaw]

            print_info(f"Using ROS Odometry for tracking. Start: {start_pose}")

            # Convert path to the global frame.
            cx, cy, cyaw = start_pose
            cos_t, sin_t = math.cos(cyaw), math.sin(cyaw)

            # Global Goal.
            gx = current_goal[0] * cos_t - current_goal[1] * sin_t + cx
            gy = current_goal[0] * sin_t + current_goal[1] * cos_t + cy
            global_goal = np.array([gx, gy, 0.0])

            # We do not need path_global initially because we will re-plan.

        start_time = time.time()
        last_replan_time = time.time()
        last_loop_time = time.time()
        max_execution_time = 60.0
        current_idx = 0

        while self.running:
            now = time.time()
            # Calculate dt for dead reckoning (covering sleep + execution time).
            dt = now - last_loop_time
            last_loop_time = now

            if now - start_time > max_execution_time:
                print_warning("Execution timeout (60s)")
                break

            # --- Robust Dead Reckoning Update ---
            # Update path and goal based on movement since the last loop.
            # This accounts for time spent in replanning/processing/sleep.
            if not use_odom and dt > 0:
                # Use the last commanded velocity (active during the interval).
                # Apply correction factors for G1 slip/lag (conservative
                # estimation). We assume actual speed is ~70% of commanded due
                # to slip/acceleration lag.
                linear_scale = 0.7
                angular_scale = 0.8

                d_dist = self.target_vx * dt * linear_scale
                d_yaw = self.target_vyaw * dt * angular_scale

                # Transform coordinate frame: translate then rotate.
                # New_P = R(-d_yaw) * (Old_P - [d_dist, 0]).
                cos_t, sin_t = math.cos(-d_yaw), math.sin(-d_yaw)

                # 1. Update Path.
                new_path = []
                for p in path:
                    # Translate (robot moved forward).
                    px_t = p[0] - d_dist
                    py_t = p[1]
                    # Rotate (robot turned).
                    px = px_t * cos_t - py_t * sin_t
                    py = px_t * sin_t + py_t * cos_t
                    new_path.append([px, py, p[2]])
                path = np.array(new_path)

                # 2. Update Goal.
                gx_t = current_goal[0] - d_dist
                gy_t = current_goal[1]
                gx = gx_t * cos_t - gy_t * sin_t
                gy = gx_t * sin_t + gy_t * cos_t
                current_goal = np.array([gx, gy, current_goal[2]])

                # Safety: Check if we passed the goal (X < 0).
                if current_goal[0] < -0.1:  # Allow slight overshoot
                    print_warning(
                        f"Goal passed (X={current_goal[0]:.2f}m < 0). Stopping."
                    )
                    break

            # --- Continuous Re-planning Check ---
            if now - last_replan_time > Config.REPLAN_INTERVAL:
                # Update camera data for re-planning.
                if self.update_camera_data():
                    rgb = self.rgb_image
                    depth = self.depth_image

                    # Calculate new local goal.
                    new_local_goal = current_goal[:2]

                    if use_odom and self.current_pose is not None:
                        # Transform global goal to the current local frame.
                        with self.odom_lock:
                            curr_pose = self.current_pose.copy()

                        dx = global_goal[0] - curr_pose[0]
                        dy = global_goal[1] - curr_pose[1]
                        cyaw = curr_pose[2]
                        cos_t, sin_t = math.cos(cyaw), math.sin(cyaw)

                        # R^T * (P_global - T).
                        lx = dx * cos_t + dy * sin_t
                        ly = -dx * sin_t + dy * cos_t
                        new_local_goal = np.array([lx, ly])

                        # Update current_goal for local checks.
                        current_goal = np.array([lx, ly, 0.0])

                    # Request new plan.
                    try:
                        new_traj, _ = self.planner_client.get_plan(
                            rgb, depth, new_local_goal
                        )
                        if new_traj is not None and len(new_traj) > 1:
                            path = np.array(new_traj)
                            current_idx = 0  # Reset index for the new path

                            # Visualization: save the re-planned trajectory.
                            timestamp = time.strftime("%H%M%S")
                            save_name = f"replan_{timestamp}.jpg"
                            target_3d_viz = (
                                new_local_goal[0],
                                new_local_goal[1],
                                0.0,
                            )

                            # Calculate pixel coordinates for target (approx).
                            target_pixel_viz = None
                            if self.fx:
                                sim_x = -new_local_goal[1]
                                sim_z = new_local_goal[0]
                                cam_height = Config.CAMERA_HEIGHT
                                roll_corr = Config.CAMERA_ROLL_CORRECTION
                                sim_y = cam_height + sim_x * roll_corr
                                if sim_z > 0.01:
                                    u = int((sim_x * self.fx) / sim_z + self.cx)
                                    v = int((sim_y * self.fy) / sim_z + self.cy)
                                    target_pixel_viz = (u, v)

                            self.project_trajectory_to_image(
                                path,
                                rgb,
                                save_name=save_name,
                                target_pixel=target_pixel_viz,
                                target_3d=target_3d_viz,
                                on_trajectory_image=on_trajectory_image,
                            )
                            # print_info(f"Re-planned trajectory with {len(path)} points")
                    except Exception as e:
                        print_warning(f"Re-planning failed: {e}")

                last_replan_time = now

            # --- Update Path using Odometry or Dead Reckoning ---
            if use_odom and self.current_pose is not None:
                with self.odom_lock:
                    curr_pose = self.current_pose.copy()

                # Update goal distance check.
                dist_to_global_goal = np.linalg.norm(
                    global_goal[:2] - curr_pose[:2]
                )

                # Note: path is already in the local frame from re-planning or
                # the previous iteration. If we rely solely on re-planning, we
                # do not need to transform the OLD path. But between re-plans
                # (0.5s), the robot moves, so the path becomes stale relative to
                # the robot. Ideally, re-planning is fast enough. If not, we
                # should technically transform 'path' based on the odometry
                # delta since the last re-plan. For a 0.5s interval, pure pursuit
                # on the last plan is acceptable, or we can assume the path is
                # static in the global frame and transform it. However, iPlanner
                # outputs a LOCAL path.

                pass  # Use the current 'path' (local) and 'current_idx'

            else:
                # Dead Reckoning Update.
                pass

            # Pure Pursuit: find the lookahead point.
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
            dist_to_goal = np.linalg.norm(current_goal[:2])

            if dist_to_goal < goal_tolerance:
                print_success(f"Reached goal area (Dist: {dist_to_goal:.2f}m)")
                break

            # Compute steering.
            # Fix: when very close to the goal (blind spot), stop yaw correction
            # to avoid drift. This prevents the "left turn" issue when the target
            # is under the robot/camera.
            if dist_to_goal < 2.0:  # Final approach blind mode (0.6m)
                cmd_yaw_speed = 0.0
                alpha = 0.0
            else:
                alpha = math.atan2(dy, dx)
                cmd_yaw_speed = np.clip(
                    (2.0 * target_speed * math.sin(alpha))
                    / max(dist_to_target, 0.1),
                    -max_yaw_speed,
                    max_yaw_speed,
                )

            # Reduce speed when turning sharply.
            current_speed = max(
                0.1, target_speed * (1.0 - abs(alpha) / math.pi)
            )

            # Clamp speeds for G1 safety.
            current_speed = min(current_speed, Config.MAX_FORWARD_SPEED)

            # Apply Yaw Bias Compensation.
            # This handles systematic drift (e.g. mechanical imbalance or
            # sensor bias).
            if hasattr(Config, "YAW_BIAS_COMPENSATION"):
                cmd_yaw_speed += Config.YAW_BIAS_COMPENSATION

            # Send command.
            self._set_velocity(current_speed, 0.0, cmd_yaw_speed)
            time.sleep(0.05)

        self.stop_robot()

    # =========================================================================
    # Visualization
    # =========================================================================

    def project_trajectory_to_image(
        self,
        trajectory_points,
        img_bgr,
        save_name="planned_path.png",
        target_pixel=None,
        target_3d=None,
        on_trajectory_image=None,
    ):
        """Project 3D trajectory points back to the 2D image and save it.

        Coordinate mapping:
        - Robot X (forward) -> Camera Z
        - Robot Y (left) -> Camera -X
        - Camera Y (down) = camera_height (ground plane assumption)
        """
        import cv2

        if trajectory_points is None or len(trajectory_points) == 0:
            return

        if self.fx is None:
            print_warning("Camera intrinsics missing, cannot draw trajectory")
            return

        vis_img = img_bgr.copy()
        h, w = vis_img.shape[:2]

        points_2d = []
        for pt in trajectory_points:
            # Robot frame to camera frame.
            sim_z = pt[0]  # Forward -> Z_cam
            sim_x = -pt[1]  # Left -> -X_cam

            cam_height = Config.CAMERA_HEIGHT
            roll_corr = Config.CAMERA_ROLL_CORRECTION
            sim_y = cam_height + sim_x * roll_corr

            if sim_z <= 0.01:
                continue

            u = int((sim_x * self.fx) / sim_z + self.cx)
            v = int((sim_y * self.fy) / sim_z + self.cy)

            if 0 <= u < w and 0 <= v < h:
                points_2d.append((u, v))

        # Draw trajectory.
        if len(points_2d) > 1:
            for i in range(len(points_2d) - 1):
                cv2.line(vis_img, points_2d[i], points_2d[i + 1], (0, 255, 0), 2)
            cv2.circle(vis_img, points_2d[-1], 6, (0, 0, 255), -1)

        # Draw target pixel marker.
        if target_pixel is not None:
            tx, ty = target_pixel
            cv2.drawMarker(
                vis_img,
                (tx, ty),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )

        # Add info text.
        if target_3d is not None:
            info_text = f"Goal: ({target_3d[0]:.2f}, {target_3d[1]:.2f})m"
            cv2.putText(
                vis_img,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            vis_img,
            "G1 Humanoid",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        save_path = os.path.join(Config.IPLANNER_DIR, save_name)
        cv2.imwrite(save_path, vis_img)
        self.iplanner_history.append(save_path)
        if on_trajectory_image is not None:
            on_trajectory_image(save_path)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def shutdown(self):
        """Clean shutdown of all robot systems."""
        print_robot("Shutting down...")
        self.running = False
        self.stop_robot()

        # Stop Orbbec pipelines.
        if hasattr(self, "pipelines") and self.pipelines:
            for position, pipeline in self.pipelines.items():
                try:
                    pipeline.stop()
                    print_info(f"Stopped {position} camera")
                except Exception as e:
                    print_error(f"Error stopping {position} camera: {e}")
            self.pipelines = {}

        print_success("Robot controller shutdown complete")
