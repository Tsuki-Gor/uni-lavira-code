"""Mapped simulator execution, independent of the navigation algorithm.

The host supplies mapping, local controllers, sensors and visualization. This
module neither reads model replies nor creates/resets/closes the environment.
"""

import logging
import os
import cv2
import numpy as np

from . import ExecutionReport, Frame, MapGoal, Panorama, PixelGoal


class MappedSimulatorRuntime:
    def __init__(self, host, observations, full_pose, *, tensor_api, actions,
                 project_pixel, displacement, collision_check, log=None):
        self.host = host
        self.tensor_api = tensor_api
        self.actions = actions
        self.project_pixel = project_pixel
        self.displacement = displacement
        self.collision_check = collision_check
        self.logger = log or logging.getLogger(__name__)
        self.obs = observations
        self.full_pose = full_pose
        self.full_map = host.mapping_module.get_full_map()
        self.current_pose = full_pose[0] if full_pose is not None else None
        self.step = 0
        self.current_step = 0
        self.action_queue = []
        self.dones = [False] * host.config.NUM_ENVIRONMENTS
        self.infos = [{}] * host.config.NUM_ENVIRONMENTS
        self.collided = 0
        self.panorama_interrupted = False
        self.reference_frame = "episode_map:" + str(host.current_episode_id)
        self.metres_per_cell = host.resolution / 100.0
        self.annotations = {}
        host._action = None

    @property
    def running(self):
        return not self.dones[0] and not self.panorama_interrupted

    @property
    def busy(self):
        return bool(self.action_queue)

    @property
    def terrain_hint(self):
        return getattr(self.host, "terrain_hint", False)

    @terrain_hint.setter
    def terrain_hint(self, value):
        self.host.terrain_hint = value

    @property
    def position(self):
        position = self.current_pose[:2] * 100 / self.host.resolution
        return self.map_goal(int(position[0]), int(position[1]))

    def map_goal(self, x, y):
        return MapGoal(x, y, self.metres_per_cell, self.reference_frame)

    def frame(self, image, depth, pose, heading=0, pose_heading_offset=None):
        return Frame(image=image, depth=depth, pose=pose, heading_degrees=heading,
                     camera_intrinsics=self.host._get_camera_intrinsics(),
                     reference_frame=self.reference_frame, color_order="RGB",
                     depth_unit="normalized_0_1", width=self.host.width, height=self.host.height,
                     pose_heading_offset=pose_heading_offset)

    def observe_current(self):
        return self.frame(self.obs[0]['rgb'], self.obs[0]['depth'], self.current_pose)

    def prepare_step(self, target, annotations):
        host = self.host
        self.annotations = annotations
        host.pos_list.append(host.envs.call_at(0, '_env')._sim.get_agent_state().position)
        host.visualizer.instruction = host.instruction
        host.visualizer.destination = host.destination
        host.visualizer._action = host._action
        self.logger.info(f"\nepisode:{host.current_episode_id}, step:{self.step}")
        host.visualizer._save_depth(self.obs[0]['depth'], self.step, use_colormap=False)
        self.current_pose = self.full_pose[0]
        self.current_step = host.current_step = self.step
        host.visualizer.sync(self.step, host.current_episode_id)
        trajectory = getattr(host, 'navdp_traj_global', None)
        trajectory = [(pt[0], pt[1]) for pt in trajectory] if trajectory is not None else None
        host.visualizer._save_rgb_frame(
            self.obs[0], self.step, annotations.get('waypoints'), host.current_episode_id,
            (target.x, target.y) if target is not None else (None, None),
            todo_list=annotations.get('todo_list'), la_output=annotations.get('la_output', {}),
            navdp_traj=trajectory)
        return self.observe_current()

    def observe_panorama(self, step):
        origin = self.observe_current()
        frames = self._scan(self.obs[0], step)
        # Compatibility: the release always charged twelve steps, even if the
        # environment finished partway through capture; its last obs stayed local.
        self.step += 12
        if 'episode_finished' in frames:
            self.panorama_interrupted = True
            return None
        views = {}
        for name, frame in zip(('forward', 'left', 'behind', 'right'), frames):
            # Keep the release's base-pose reference. A CPU map's NumPy view
            # may mutate; freezing it here would silently change backtracking.
            views[name] = self.frame(frame['rgb'], frame['depth'], origin.pose, frame['angle'], frame['angle'])
        return Panorama(views, origin=origin)

    def record_panorama(self, frames, waypoint):
        self.host.visualizer._save_waypoint_panorama_rgb(frames, waypoint, self.step)

    def turn(self, degrees):
        if degrees not in (0, 90, -90, 180):
            raise ValueError("this runtime supports 0, +/-90 or 180 degree turns")
        action = self.actions.TURN_LEFT if degrees > 0 else self.actions.TURN_RIGHT
        self.action_queue.extend([action] * int(abs(degrees) / 30))

    def stop(self):
        self.action_queue.append(0)  # Release STOP action in the simulator action space.

    def project(self, goal, adjust_depth=False, fallback=None):
        if not isinstance(goal, PixelGoal):
            raise TypeError("projection requires a PixelGoal bound to a captured view")
        view = goal.view
        if view.reference_frame != self.reference_frame or view.depth_unit != "normalized_0_1":
            raise ValueError("view must use this episode map and normalized simulator depth")
        depth = view.depth
        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=2)
        depth = self.host._preprocess_depth(depth, 0.1, 5.0) / 100.0
        if view.pose is None:
            if fallback is None:
                raise ValueError("capture pose or explicit map fallback is required")
            return fallback
        pose = view.pose
        if view.pose_heading_offset is not None:
            pose = pose.clone().cpu().numpy() if isinstance(pose, self.tensor_api.Tensor) else pose.copy()
            if pose.ndim == 2:
                pose = pose[0]
            pose[2] = (pose[2] + view.pose_heading_offset) % 360
        while True:
            target = self.project_pixel(pixel_coords=(goal.u, goal.v), depth_image=depth,
                                        full_pose=pose, camera_intrinsics=view.camera_intrinsics)
            x = max(0, min(int(target[0] * 100.0 / self.host.resolution), self.host.map_shape[0] - 1))
            y = max(0, min(int(target[1] * 100.0 / self.host.resolution), self.host.map_shape[1] - 1))
            if not adjust_depth or self.host.traversable[y, x] == 1 or depth.max() < 0.1:
                return self.map_goal(x, y)
            depth = depth - 0.1

    def queue_navigation(self, goal, initial=False):
        if goal.reference_frame != self.reference_frame or goal.metres_per_cell != self.metres_per_cell:
            raise ValueError("goal must use this episode's map reference and scale")
        action = self._plan(goal, initial)
        if action is not None:
            self.action_queue.append(action)

    def execute_pending(self):
        if not self.action_queue:
            return None
        host = self.host
        host._action = self.action_queue.pop(0)
        outputs = host.envs.step([{"action": host._action}])
        self.step += 1
        self.obs, _, self.dones, self.infos = [list(x) for x in zip(*outputs)]
        if not self.dones[0]:
            batch_obs = host._batch_obs(self.obs)
            poses = self.tensor_api.from_numpy(np.array([item['sensor_pose'] for item in self.obs])).float().to(host.device)
            host.mapping_module(batch_obs, poses, self.current_step)
            self.full_map, self.full_pose, one_step_full_map = host.mapping_module.update_map(
                self.step, host.detected_classes, host.current_episode_id)
            host.mapping_module.one_step_full_map.fill_(0.)
            host.mapping_module.one_step_local_map.fill_(0.)
            host.traversable, host.floor, host.frontiers = host._process_map(self.step, self.full_map[0])
            host.one_step_floor = host._process_one_step_floor(one_step_full_map[0])
            last_pose = self.current_pose
            self.current_pose = self.full_pose[0]
            if last_pose is not None and self.current_pose is not None:
                distance = self.displacement(last_pose, self.current_pose, host.resolution)
                self.collided = self.collided + 1 if distance < 0.2 * 100 / host.resolution else 0
                if self.collided >= 30:
                    fname = os.path.join(host.config.EVAL_CKPT_PATH_DIR,
                                         f"r{host.local_rank}_w{host.world_size}_collision_stuck.txt")
                    with open(fname, "a") as stream:
                        stream.writelines(f"id: {str(host.current_episode_id)}; step: {str(self.step)}; collided: {str(self.collided)}\n")
            if last_pose is not None and host._action == 1:
                collision_map = self.collision_check(last_pose, self.current_pose, host.resolution, host.mapping_module.map_shape)
                host.collision_map = np.logical_or(host.collision_map, collision_map)
            host.traversable[host.collision_map == 1] = 0
        return ExecutionReport(execution_called=True)

    def _plan(self, goal, initial):
        torch, logger, HabitatSimActions = self.tensor_api, self.logger, self.actions
        current_pose, obs, full_map, step = self.current_pose, self.obs, self.full_map, self.step
        target_map_x, target_map_y = goal.x, goal.y
        terrain_hint, annotations = self.terrain_hint, self.annotations
        action_list = []
        search_destination = False
        if initial:
            while True:
                waypoint = np.array([target_map_y, target_map_x])

                # Controller Selection Logic (Step 2)
                use_special_controller = False
                selected_controller = None # 'iplanner', 'navdp'

                # Determine if we should use a special controller (NavDP/iPlanner)
                # Logic:
                # 1. if use_fmm is False -> MUST use special controller (NavDP or iPlanner, whichever is True)
                # 2. if use_fmm is True -> use special controller ONLY IF (stair is True AND one of them is True)

                has_special_controller = self.host.use_navdp or self.host.use_iplanner

                if not self.host.use_fmm:
                    # FMM disabled, must use special controller
                    if self.host.use_iplanner: selected_controller = 'iplanner'
                    elif self.host.use_navdp: selected_controller = 'navdp'
                    use_special_controller = True
                else:
                    # FMM enabled
                    if has_special_controller and terrain_hint:
                         # Hybrid mode: Use special controller for stairs
                         if self.host.use_iplanner: selected_controller = 'iplanner'
                         elif self.host.use_navdp: selected_controller = 'navdp'
                         use_special_controller = True
                    else:
                         # Use FMM (Normal case or stairs but no special controller enabled)
                         use_special_controller = False

                if use_special_controller:
                    traj_np = None  # Initialize to avoid undefined variable error
                    if selected_controller == 'iplanner':
                        logger.info(f"DEBUG: Step 2 - Using iPlanner (stair={terrain_hint})")
                        # Calculate local goal from target_map_x, target_map_y
                        # Convert map indices to world meters
                        tx = target_map_x * self.host.resolution / 100.0
                        ty = target_map_y * self.host.resolution / 100.0

                        ax = current_pose[0] # x in meters
                        ay = current_pose[1] # y in meters (z in world)
                        ah = np.deg2rad(current_pose[2]) # heading in radians (converted from degrees)

                        # Relative vector in world frame
                        dx = tx - ax
                        dy = ty - ay

                        # Rotate to robot frame (X-forward, Y-left, Z-up)
                        x_body = dx * np.cos(ah) + dy * np.sin(ah)   # Forward distance
                        y_body = -dx * np.sin(ah) + dy * np.cos(ah)  # Left distance
                        z_body = 0.0                                 # Up distance

                        local_goal = torch.tensor([[x_body, y_body, z_body]]).to(self.host.device)

                        # Call iplanner (Force update)
                        min_d = self.host.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                        max_d = self.host.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                        depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)

                        dep_tensor = torch.from_numpy(depth_meters).to(self.host.device).unsqueeze(0) # (1, H, W, 1)
                        keypoints, traj, fear = self.host.iplanner_agent.step_pointgoal(dep_tensor, local_goal)
                        traj_np = None  # Fix: initialize to avoid undefined variable

                        # Store Global Trajectory
                        traj_np = traj[0].cpu().numpy()
                        global_traj = []
                        for pt in traj_np:
                            xb, yb, zb = pt
                            w_dx = xb * np.cos(ah) - yb * np.sin(ah)
                            w_dy = xb * np.sin(ah) + yb * np.cos(ah)
                            wx = ax + w_dx
                            wy = ay + w_dy
                            global_traj.append([wx, wy, zb])
                        self.host.navdp_traj_global = np.array(global_traj)
                        self.host.last_iplanner_step = step

                        if traj_np is not None:  # Check if traj_np is defined
                            # Visualization (Project traj_np to depth)
                            if self.host.visualize:
                                try:
                                    # Use traj_np (Local Frame)
                                    viz_depth = self.host.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                                    if viz_depth is not None:
                                        if len(viz_depth.shape) == 2:
                                            viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                        elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                            viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)

                                        intrinsics = self.host._get_camera_intrinsics()
                                        fx, fy = intrinsics[0,0], intrinsics[1,1]
                                        cx, cy = intrinsics[0,2], intrinsics[1,2]
                                        h, w = viz_depth.shape[:2]

                                        valid_points = []
                                        for pt in traj_np:
                                            x, y, z = pt # x=forward, y=left, z=up

                                            xc = -y
                                            yc = -z
                                            zc = x

                                            if x > 0.1:
                                                u = fx * (-y) / x + cx
                                                u = int(u)
                                                v = int(fy * (self.host.config.MAP.AGENT_HEIGHT - z) / x + cy)

                                                if 0 <= u < w and 0 <= v < h:
                                                    valid_points.append((u, v))

                                        # Draw lines
                                        for i in range(len(valid_points)-1):
                                            cv2.line(viz_depth, valid_points[i], valid_points[i+1], (0, 0, 255), 2) # Red
                                        for pt in valid_points:
                                            cv2.circle(viz_depth, pt, 3, (0, 255, 0), -1) # Green

                                        # Save
                                        episode_dir = os.path.join(self.host.save_dir, str(self.host.current_episode_id))
                                        os.makedirs(episode_dir, exist_ok=True)
                                        cv2.imwrite(os.path.join(episode_dir, f"navdp_traj_step{step}.png"), viz_depth)
                                except Exception as e:
                                    logger.info(f"Error visualizing iplanner: {e}")

                            # Direct Control Action
                            # Find target point
                            lookahead_dist = 1.0
                            target_pt = None
                            for pt in traj_np:
                                dist = np.linalg.norm(pt[:2]) # x, y
                                if dist > lookahead_dist:
                                    target_pt = pt
                                    break
                            if target_pt is None and len(traj_np) > 0:
                                target_pt = traj_np[-1]

                            action_id = HabitatSimActions.MOVE_FORWARD
                            if target_pt is not None:
                                x, y = target_pt[:2]
                                angle = np.arctan2(y, x) # y is left, x is forward. +angle = left.
                                # Threshold 15 deg = 0.26 rad
                                if angle > 0.26:
                                    action_id = HabitatSimActions.TURN_LEFT
                                elif angle < -0.26:
                                    action_id = HabitatSimActions.TURN_RIGHT
                                else:
                                    action_id = HabitatSimActions.MOVE_FORWARD

                            action_list.append(action_id)
                            break # Skip FMM
                    elif selected_controller == 'navdp':
                        # logger.info(f"DEBUG: Step 2 - Using NavDP (stair={terrain_hint})")
                        tx = target_map_x * self.host.resolution / 100.0
                        ty = target_map_y * self.host.resolution / 100.0
                        ax = current_pose[0]
                        ay = current_pose[1]
                        ah = np.deg2rad(current_pose[2])

                        dx = tx - ax
                        dy = ty - ay

                        x_body = dx * np.cos(ah) + dy * np.sin(ah)
                        y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                        z_body = 0.0

                        goals = np.array([[x_body, y_body, z_body]])
                        rgb_img = obs[0]['rgb']
                        min_d = self.host.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                        max_d = self.host.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                        depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                        depth_img = depth_meters

                        if depth_img.ndim == 2:
                            depth_img = depth_img[:, :, np.newaxis]
                        images = rgb_img[np.newaxis, ...]
                        depths = depth_img[np.newaxis, ...]

                        next_point, traj, val, mask = self.host.navdp_agent.step_pointgoal(goals, images, depths)

                        # Visualization
                        local_traj = next_point[0]
                        global_traj = []
                        for pt in local_traj:
                            lb_x, lb_y = pt[:2]
                            wx = ax + lb_x * np.cos(ah) - lb_y * np.sin(ah)
                            wy = ay + lb_x * np.sin(ah) + lb_y * np.cos(ah)
                            global_traj.append([wx, wy])
                        self.host.navdp_traj_global = np.array(global_traj)

                        x, y = local_traj[0][:2]
                        angle = np.arctan2(y, x)
                        if angle > 0.26:
                            action_id = HabitatSimActions.TURN_LEFT
                        elif angle < -0.26:
                            action_id = HabitatSimActions.TURN_RIGHT
                        else:
                            action_id = HabitatSimActions.MOVE_FORWARD

                        action_list.append(action_id)
                        break


                # Emit the first navigation action immediately so the agent doesn't idle.
                if not use_special_controller and self.host.use_fmm:
                    # Add Visualization for FMM
                    if self.host.visualize:
                        try:
                            viz_depth = self.host.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                            if viz_depth is not None:
                                if len(viz_depth.shape) == 2:
                                    viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                                elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                    viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)

                                intrinsics = self.host._get_camera_intrinsics()
                                fx, fy = intrinsics[0,0], intrinsics[1,1]
                                cx, cy = intrinsics[0,2], intrinsics[1,2]
                                h, w = viz_depth.shape[:2]

                                # Draw Goal
                                ax = current_pose[0]
                                ay = current_pose[1]
                                ah = np.deg2rad(current_pose[2])

                                tx_viz = target_map_x * self.host.resolution / 100.0
                                ty_viz = target_map_y * self.host.resolution / 100.0
                                dx_viz = tx_viz - ax
                                dy_viz = ty_viz - ay
                                gx = dx_viz * np.cos(ah) + dy_viz * np.sin(ah)
                                gy = -dx_viz * np.sin(ah) + dy_viz * np.cos(ah)
                                gz = 0.0

                                if gx > 0.1:
                                    gu = fx * (-gy) / gx + cx
                                    gv = fy * (self.host.config.MAP.AGENT_HEIGHT - gz) / gx + cy
                                    gu, gv = int(gu), int(gv)
                                    if 0 <= gu < w and 0 <= gv < h:
                                        # Draw Yellow Star for Goal
                                        cv2.drawMarker(viz_depth, (gu, gv), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                                        cv2.putText(viz_depth, "Goal", (gu + 10, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                                episode_dir = os.path.join(self.host.save_dir, str(self.host.current_episode_id))
                                os.makedirs(episode_dir, exist_ok=True)
                                cv2.imwrite(os.path.join(episode_dir, f"navdp_traj_step{step}.png"), viz_depth)
                        except Exception as e:
                            logger.info(f"Error in FMM viz: {e}")

                    navigation_action = self.host.policy._get_action(
                        current_pose, waypoint, full_map[0], self.host.traversable,
                        self.host.collision_map, step, self.host.current_episode_id,
                        self.host.detected_classes, search_destination
                    )
                    action_list.append(navigation_action)
                elif not use_special_controller and not self.host.use_fmm:
                    logger.warning("Step 2 - FMM is disabled and no other planner took over! This should not happen due to init validation.")
                break
        else:
            waypoint = np.array([target_map_y, target_map_x])

            # Controller Selection Logic (Step 3)
            use_special_controller = False
            selected_controller = None

            has_special_controller = self.host.use_navdp or self.host.use_iplanner

            if not self.host.use_fmm:
                if self.host.use_iplanner: selected_controller = 'iplanner'
                elif self.host.use_navdp: selected_controller = 'navdp'
                use_special_controller = True
            else:
                if has_special_controller and terrain_hint:
                     if self.host.use_iplanner: selected_controller = 'iplanner'
                     elif self.host.use_navdp: selected_controller = 'navdp'
                     use_special_controller = True
                else:
                     use_special_controller = False

            if selected_controller == 'iplanner':
                traj_local_np = None

                ax = current_pose[0]
                ay = current_pose[1]
                ah = np.deg2rad(current_pose[2])

                if step - self.host.last_iplanner_step >= self.host.iplanner_update_freq or self.host.navdp_traj_global is None:
                    # Replan
                    tx = target_map_x * self.host.resolution / 100.0
                    ty = target_map_y * self.host.resolution / 100.0

                    dx = tx - ax
                    dy = ty - ay

                    x_body = dx * np.cos(ah) + dy * np.sin(ah)
                    y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                    z_body = 0.0

                    local_goal = torch.tensor([[x_body, y_body, z_body]]).to(self.host.device)

                    min_d = self.host.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                    max_d = self.host.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                    depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                    dep_tensor = torch.from_numpy(depth_meters).to(self.host.device).unsqueeze(0)

                    keypoints, traj, fear = self.host.iplanner_agent.step_pointgoal(dep_tensor, local_goal)

                    # Update Global
                    traj_local_np = traj[0].cpu().numpy()
                    global_traj = []
                    for pt in traj_local_np:
                        xb, yb, zb = pt
                        w_dx = xb * np.cos(ah) - yb * np.sin(ah)
                        w_dy = xb * np.sin(ah) + yb * np.cos(ah)
                        wx = ax + w_dx
                        wy = ay + w_dy
                        global_traj.append([wx, wy, zb])
                    self.host.navdp_traj_global = np.array(global_traj)
                    self.host.last_iplanner_step = step
                    logger.info(f"DEBUG: iPlanner generated trajectory with {len(global_traj)} points in Step 3")
                else:
                    # Use stored global traj -> local
                    if self.host.navdp_traj_global is not None and len(self.host.navdp_traj_global) > 0:
                        local_traj = []
                        for gpt in self.host.navdp_traj_global:
                            gx, gy, gz = gpt
                            dx = gx - ax
                            dy = gy - ay
                            # Inverse rotate
                            xb = dx * np.cos(ah) + dy * np.sin(ah)
                            yb = -dx * np.sin(ah) + dy * np.cos(ah)
                            local_traj.append([xb, yb, gz])
                        traj_local_np = np.array(local_traj)
                    else:
                        traj_local_np = np.array([])

                # Unified Visualization on VLM Map (Update current frame)
                if self.host.visualize and self.host.navdp_traj_global is not None:
                    try:
                        traj_world_list = [(pt[0], pt[1]) for pt in self.host.navdp_traj_global]
                        self.host.visualizer._save_rgb_frame(
                            obs[0],
                            step,
                            annotations.get('waypoints', []),
                            self.host.current_episode_id,
                            (target_map_x, target_map_y),
                            todo_list=annotations.get('todo_list'),
                            navdp_traj=traj_world_list
                        )
                    except Exception as e:
                        logger.info(f"Error updating VLM map viz in Step 3: {e}")

                # Visualization (Project traj_local_np to depth)
                if self.host.visualize:
                    try:
                        viz_depth = self.host.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                        if viz_depth is not None:
                            if len(viz_depth.shape) == 2:
                                viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                            elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)

                            intrinsics = self.host._get_camera_intrinsics()
                            fx, fy = intrinsics[0,0], intrinsics[1,1]
                            cx, cy = intrinsics[0,2], intrinsics[1,2]
                            h, w = viz_depth.shape[:2]

                            valid_points = []
                            if traj_local_np is not None:
                                for pt in traj_local_np:
                                    x, y, z = pt
                                    if x > 0.1:
                                        u = fx * (-y) / x + cx
                                        v = fy * (self.host.config.MAP.AGENT_HEIGHT - z) / x + cy
                                        u = int(u)
                                        v = int(v)
                                        if 0 <= u < w and 0 <= v < h:
                                            valid_points.append((u, v))

                            for i in range(len(valid_points)-1):
                                cv2.line(viz_depth, valid_points[i], valid_points[i+1], (0, 0, 255), 2)
                            for pt in valid_points:
                                cv2.circle(viz_depth, pt, 3, (0, 255, 0), -1)

                            # Draw Goal
                            # Recalculate local goal for visualization
                            tx_viz = target_map_x * self.host.resolution / 100.0
                            ty_viz = target_map_y * self.host.resolution / 100.0
                            dx_viz = tx_viz - ax
                            dy_viz = ty_viz - ay
                            gx = dx_viz * np.cos(ah) + dy_viz * np.sin(ah)
                            gy = -dx_viz * np.sin(ah) + dy_viz * np.cos(ah)
                            gz = 0.0

                            if gx > 0.1:
                                gu = fx * (-gy) / gx + cx
                                gv = fy * (self.host.config.MAP.AGENT_HEIGHT - gz) / gx + cy
                                gu, gv = int(gu), int(gv)
                                if 0 <= gu < w and 0 <= gv < h:
                                    # Draw Yellow Star for Goal
                                    cv2.drawMarker(viz_depth, (gu, gv), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                                    cv2.putText(viz_depth, "Goal", (gu + 10, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                            episode_dir = os.path.join(self.host.save_dir, str(self.host.current_episode_id))
                            os.makedirs(episode_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(episode_dir, f"navdp_traj_step{step}.png"), viz_depth)
                    except Exception as e:
                        logger.info(f"Error in iplanner viz: {e}")

                # Action
                lookahead_dist = 0.1
                target_pt = None
                if traj_local_np is not None:
                    for pt in traj_local_np:
                        dist = np.linalg.norm(pt[:2])
                        if dist > lookahead_dist:
                            target_pt = pt
                            break
                    if target_pt is None and len(traj_local_np) > 0:
                        target_pt = traj_local_np[-1]

                action_id = HabitatSimActions.MOVE_FORWARD
                if target_pt is not None:
                    x, y = target_pt[:2]
                    angle = np.arctan2(y, x)
                    if angle > 0.26:
                        action_id = HabitatSimActions.TURN_LEFT
                    elif angle < -0.26:
                        action_id = HabitatSimActions.TURN_RIGHT
                    else:
                        action_id = HabitatSimActions.MOVE_FORWARD

                action_list.append(action_id)
            elif selected_controller == 'navdp':
                tx = target_map_x * self.host.resolution / 100.0
                ty = target_map_y * self.host.resolution / 100.0
                ax = current_pose[0]
                ay = current_pose[1]
                ah = np.deg2rad(current_pose[2])

                dx = tx - ax
                dy = ty - ay
                x_body = dx * np.cos(ah) + dy * np.sin(ah)
                y_body = -dx * np.sin(ah) + dy * np.cos(ah)
                z_body = 0.0

                goals = np.array([[x_body, y_body, z_body]])
                rgb_img = obs[0]['rgb']
                min_d = self.host.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
                max_d = self.host.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
                depth_meters = min_d + obs[0]['depth'] * (max_d - min_d)
                depth_img = depth_meters
                # NavDP Frequency Control
                # Initialize variables if not present
                if not hasattr(self, 'last_navdp_step'):
                    self.host.last_navdp_step = -999
                if not hasattr(self, 'navdp_local_traj'):
                    self.host.navdp_local_traj = None

                # Execute NavDP planning every 5 steps or if no trajectory exists
                if step - self.host.last_navdp_step >= 5 or self.host.navdp_local_traj is None or len(self.host.navdp_local_traj) == 0:
                    # logger.info(f"DEBUG: Executing NavDP planning at step {step}")

                    if depth_img.ndim == 2:
                        depth_img = depth_img[:, :, np.newaxis]
                    images = rgb_img[np.newaxis, ...]
                    depths = depth_img[np.newaxis, ...]

                    next_point, traj, val, mask = self.host.navdp_agent.step_pointgoal(goals, images, depths)

                    # Store the new trajectory and update timestamp
                    self.host.navdp_local_traj = next_point[0] # (N, 3)
                    self.host.last_navdp_step = step

                    # Update Global Visualization (Optional, but good for debugging)
                    local_traj = self.host.navdp_local_traj
                    global_traj = []
                    for pt in local_traj:
                        lb_x, lb_y = pt[:2]
                        wx = ax + lb_x * np.cos(ah) - lb_y * np.sin(ah)
                        wy = ay + lb_x * np.sin(ah) + lb_y * np.cos(ah)
                        global_traj.append([wx, wy])
                    self.host.navdp_traj_global = np.array(global_traj)

                # Let's use the GLOBAL trajectory method for stability across steps.
                if self.host.navdp_traj_global is not None and len(self.host.navdp_traj_global) > 0:
                    # Find a lookahead point on the global trajectory relative to CURRENT pose
                    target_pt = None
                    lookahead_dist = 0.5

                    # Convert global trajectory back to CURRENT local frame
                    current_local_traj = []
                    for gpt in self.host.navdp_traj_global:
                        gx, gy = gpt
                        dx = gx - ax
                        dy = gy - ay
                        # Inverse rotate to get local x, y
                        lx = dx * np.cos(ah) + dy * np.sin(ah)
                        ly = -dx * np.sin(ah) + dy * np.cos(ah)
                        current_local_traj.append([lx, ly])

                    # Now find the target point in current local frame
                    for pt in current_local_traj:
                        dist = np.linalg.norm(pt[:2])
                        if dist > lookahead_dist:
                            target_pt = pt
                            break

                    if target_pt is None and len(current_local_traj) > 0:
                        target_pt = current_local_traj[-1]
                else:
                    # Fallback
                    target_pt = None

                if target_pt is not None:
                    x, y = target_pt[:2]
                    angle = np.arctan2(y, x)
                    if angle > 0.26:
                        action_id = HabitatSimActions.TURN_LEFT
                    elif angle < -0.26:
                        action_id = HabitatSimActions.TURN_RIGHT
                    else:
                        action_id = HabitatSimActions.MOVE_FORWARD
                else:
                    action_id = HabitatSimActions.MOVE_FORWARD

                action_list.append(action_id)
            elif not use_special_controller and self.host.use_fmm:
                logger.info(f"DEBUG: Step 3 - Using FMM. use_iplanner={self.host.use_iplanner}, use_navdp={self.host.use_navdp}, stair={terrain_hint}")
                # Clear special controller trajectories to avoid stale visualization
                self.host.navdp_traj_global = None
                self.host.navdp_local_traj = None

                # Add Visualization for FMM
                if self.host.visualize:
                    try:
                        viz_depth = self.host.visualizer._save_depth(obs[0]['depth'], step, use_colormap=False)
                        if viz_depth is not None:
                            if len(viz_depth.shape) == 2:
                                viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)
                            elif len(viz_depth.shape) == 3 and viz_depth.shape[2] == 1:
                                viz_depth = cv2.cvtColor(viz_depth, cv2.COLOR_GRAY2BGR)

                            intrinsics = self.host._get_camera_intrinsics()
                            fx, fy = intrinsics[0,0], intrinsics[1,1]
                            cx, cy = intrinsics[0,2], intrinsics[1,2]
                            h, w = viz_depth.shape[:2]

                            # Draw Goal
                            ax = current_pose[0]
                            ay = current_pose[1]
                            ah = np.deg2rad(current_pose[2])

                            tx_viz = target_map_x * self.host.resolution / 100.0
                            ty_viz = target_map_y * self.host.resolution / 100.0
                            dx_viz = tx_viz - ax
                            dy_viz = ty_viz - ay
                            gx = dx_viz * np.cos(ah) + dy_viz * np.sin(ah)
                            gy = -dx_viz * np.sin(ah) + dy_viz * np.cos(ah)
                            gz = 0.0

                            if gx > 0.1:
                                gu = fx * (-gy) / gx + cx
                                gv = fy * (self.host.config.MAP.AGENT_HEIGHT - gz) / gx + cy
                                gu, gv = int(gu), int(gv)
                                if 0 <= gu < w and 0 <= gv < h:
                                    # Draw Yellow Star for Goal
                                    cv2.drawMarker(viz_depth, (gu, gv), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)
                                    cv2.putText(viz_depth, "Goal", (gu + 10, gv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                            episode_dir = os.path.join(self.host.save_dir, str(self.host.current_episode_id))
                            os.makedirs(episode_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(episode_dir, f"navdp_traj_step{step}.png"), viz_depth)
                    except Exception as e:
                        logger.info(f"Error in FMM viz: {e}")

                navigation_action = self.host.policy._get_action(
                    current_pose, waypoint, full_map[0], self.host.traversable,
                    self.host.collision_map, step, self.host.current_episode_id,
                    self.host.detected_classes, search_destination
                )
                action_list.append(navigation_action)
            else:
                logger.warning("Step 3 - FMM is disabled and no other planner took over!")
        return action_list[0] if action_list else None

    def _scan(self, obs, step):
        """Collect a panorama by turning left in 12 steps of 30 degrees each."""

        torch, HabitatSimActions = self.tensor_api, self.actions
        panorama_frames = []

        for turn_step in range(1, 12 + 1):
            turn_action = [{"action": HabitatSimActions.TURN_LEFT}]  # 30 deg per step
            turn_outputs = self.host.envs.step(turn_action)
            turn_obs, _, turn_dones, turn_infos = [list(x) for x in zip(*turn_outputs)]

            if turn_dones[0]:
                # Return signal that episode is done - caller should handle this
                return {'turn_direction': 'episode_done', 'episode_finished': True}

            panorama_frames.append({
                'rgb': turn_obs[0]['rgb'].copy(),
                'depth': turn_obs[0]['depth'].copy(),
                'angle': turn_step * 30 % 360,
                'step': turn_step
            })

            # Update map state for consistency (not persisted).
            batch_obs = self.host._batch_obs(turn_obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in turn_obs])).float().to(self.host.device)
            self.host.mapping_module(batch_obs, poses, self.host.current_step)
            full_map, full_pose, one_step_full_map = \
                self.host.mapping_module.update_map(step + turn_step, self.host.detected_classes, self.host.current_episode_id)
            self.host.mapping_module.one_step_full_map.fill_(0.)
            self.host.mapping_module.one_step_local_map.fill_(0.)
        panorama_frames = [panorama_frames[-1]] + panorama_frames[:-1]

        return panorama_frames[::3]
