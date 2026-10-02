"""Mapped navigation's per-step decisions; the shared agent owns the task loop."""

import logging
import numpy as np
from PIL import Image

from navigation_runtime import PixelGoal


class MappedTask:
    def __init__(self, reasoning, task, state, *, use_continuous_history=True,
                 history_interval=2, target_reached_cells=15):
        self.reasoning, self.task, self.state = reasoning, task, state
        self.instruction, self.current_episode_id = task.instruction, task.id
        self.task_type = {"vln": "VLN", "object_nav": "OBJNAV", "eqa": "EQA"}[task.kind]
        self.use_continuous_history = use_continuous_history
        self.history_interval = history_interval
        self.target_reached_threshold = target_reached_cells
        self.logger = getattr(reasoning, 'logger', logging.getLogger(__name__))
        self.going_to_stop = self.panorama_got = self.navigate_or_not = False
        self.is_backtracking = self.backtracking_only = False
        self.target_map_x = self.target_map_y = self.target_set_step = None
        self.current_la_action = 'NAVIGATE'
        self.latest_la_output = {}
        self.stop_feedback = ''
        self.consecutive_stop_failures = 0
        self.stop_requested = self._eqa_done_requested = False
        self.last_submitted_answer = None
        self.answer_error = None
        self.last_execution = None
        self.reason = 'environment_finished'

    def target(self, runtime):
        if self.target_map_x is not None and self.target_map_y is not None:
            return runtime.map_goal(self.target_map_x, self.target_map_y)
        return None

    def _capture(self, runtime):
        self._panorama = runtime.observe_panorama(runtime.step)
        if self._panorama is None:
            return None
        return [{"rgb": view.image, "depth": view.depth, "angle": view.heading_degrees,
                 "step": int(view.heading_degrees / 30) or 12}
                for view in self._panorama.views.values()]

    def _project(self, runtime, bbox, view, adjust_depth=False, fallback=None):
        runtime.terrain_hint = self.reasoning.stair
        u = int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0)
        # Release behavior: any truthy stair hint, including 'down', uses y1.
        v = int(bbox.get('y1', 0)) if self.reasoning.stair else min(int(bbox.get('y2', 0)), view.height - 1)
        return runtime.project(PixelGoal(view, u, v), adjust_depth, fallback)

    def advance(self, runtime, progress):
        progress('observing')
        view = runtime.prepare_step(self.target(runtime), {
            'waypoints': self.state.visited_targets,
            'todo_list': self.reasoning._format_todo_for_prompt() if self.reasoning.use_todo_list else None,
            'la_output': self.latest_la_output,
        })
        position = runtime.position
        if self.use_continuous_history and runtime.step % self.history_interval == 0:
            rgb = view.image.copy()
            if isinstance(rgb, np.ndarray):
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255).astype(np.uint8)
                rgb = Image.fromarray(rgb)
            self.state.history_images.append({'step': runtime.step, 'image': rgb})
        progress('reasoning')
        if not runtime.busy:
            # The release checks timeout before arrival and may discard the
            # most recent waypoint in either branch.
            if self.target_map_x is not None and self.target_map_y is not None and (self.target_set_step is not None):
                steps_since_target_set = runtime.step - self.target_set_step
                if steps_since_target_set >= 15:
                    if self.backtracking_only:
                        self.logger.info('Backtrack-only return timed out, resetting navigation state.')
                    if len(self.state.visited_targets) > 0:
                        self.state.visited_targets.pop()
                    self.panorama_got = False
                    self.navigate_or_not = False
                    (self.target_map_x, self.target_map_y) = (None, None)
                    self.target_set_step = None
                    self.backtracking_only = False
            if self.target_map_x is not None and self.target_map_y is not None:
                distance_to_target = np.sqrt((self.target_map_x - position.x) ** 2 + (self.target_map_y - position.y) ** 2)
                current_threshold = self.target_reached_threshold
                if distance_to_target < current_threshold:
                    if self.backtracking_only:
                        self.logger.info('Reached backtrack waypoint. Returning to normal decision flow.')
                    if len(self.state.visited_targets) > 0:
                        dist_calc = lambda target: np.sqrt((target['world_coords'][0] - self.state.visited_targets[-1]['world_coords'][0]) ** 2 + (target['world_coords'][1] - self.state.visited_targets[-1]['world_coords'][1]) ** 2) if 'world_coords' in target else float('inf')
                        for target in self.state.visited_targets[:-1]:
                            if dist_calc(target) < self.target_reached_threshold:
                                self.state.visited_targets.pop()
                                break
                    self.panorama_got = False
                    self.navigate_or_not = False
                    (self.target_map_x, self.target_map_y) = (None, None)
                    self.target_set_step = None
                    self.is_backtracking = False
                    self.backtracking_only = False
                    self.current_la_action = 'NAVIGATE'
            if self.target_map_x is None and (not self.panorama_got) and self.going_to_stop:
                panorama_frames = self._capture(runtime)
                if panorama_frames is None:
                    return True
                should_stop, stop_response = self.reasoning.double_check_stop(
                    self.instruction, panorama_frames, self.state.visited_targets,
                    episode_id=self.current_episode_id, step=runtime.step,
                    history_images=self.state.history_images)
                self.latest_la_output = stop_response
                if should_stop:
                    runtime.stop()
                    self.stop_requested = True
                    self.consecutive_stop_failures = 0
                    if self.task_type == 'EQA':
                        progress('answering')
                        try:
                            self.last_submitted_answer = self.reasoning.query_llm_oracle(panorama_frames, self.instruction)
                        except Exception as e:
                            self.answer_error = str(e)
                            self.logger.error(f'EQA oracle failed: {e}')
                            self.last_submitted_answer = None
                        self._eqa_done_requested = True
                else:
                    self.going_to_stop = False
                    self.navigate_or_not = False
                    self.panorama_got = False
                    self.current_la_action = 'NAVIGATE'
                    self.consecutive_stop_failures += 1
                    self.stop_feedback = f'System: Your previous decision to STOP was rejected by the double-check mechanism because the target was not clearly visible or not close enough (Failure count: {self.consecutive_stop_failures}). Please navigate closer to the target or verify your position.'
                    if self.consecutive_stop_failures >= 3:
                        self.logger.info(f'Forcing STOP after {self.consecutive_stop_failures} consecutive double-check rejections.')
                        runtime.stop()
                        self.stop_requested = True
                        self.stop_feedback = ''
                        self.consecutive_stop_failures = 0
                        if self.task_type == 'EQA':
                            progress('answering')
                            try:
                                self.last_submitted_answer = self.reasoning.query_llm_oracle(panorama_frames, self.instruction)
                            except Exception as e:
                                self.answer_error = str(e)
                                self.logger.error(f'EQA oracle failed: {e}')
                                self.last_submitted_answer = None
                            self._eqa_done_requested = True
            elif self.target_map_x is None and (not self.navigate_or_not):
                current_rgb = view.image.copy()
                panorama_frames = self._capture(runtime)
                if panorama_frames is None:
                    return True
                waypoint_id = len(self.state.visited_targets)
                self.state.visited_targets.append({
                    'step': runtime.step,
                    'init_image': Image.fromarray(current_rgb) if isinstance(current_rgb, np.ndarray) else current_rgb,
                    'panorama_frames': panorama_frames,
                    'world_coords': (position.x, position.y), 'full_pose': view.pose,
                    '_views': self._panorama.views,
                })
                self.state.total_waypoints += 1
                runtime.record_panorama(panorama_frames, waypoint_id)
                combined_feedback = self.stop_feedback
                todo_feedback = getattr(self.reasoning, 'todo_verification_feedback', '')
                if todo_feedback:
                    combined_feedback = (combined_feedback + '\n' + todo_feedback).strip()
                    self.reasoning.todo_verification_feedback = ''
                self.state.cycles += 1
                decision = self.reasoning.navigate_or_backtrack(
                    instruction=self.instruction, visited_targets=self.state.visited_targets,
                    feedback=combined_feedback, episode_id=self.current_episode_id,
                    step=runtime.step,
                    history_images=self.state.history_images if self.use_continuous_history else None)
                self.latest_la_output = decision
                if decision.get('action') == 'NAVIGATE' or decision.get('action') == 'BACKTRACK':
                    self.stop_feedback = ''
                if decision.get('action') == 'STOP':
                    self.current_la_action = 'STOP'
                if decision.get('action', 'NAVIGATE') == 'BACKTRACK':
                    self.state.total_backtracks += 1
                    target_waypoint_id = decision.get('waypoint', 0)
                    if isinstance(target_waypoint_id, int) and target_waypoint_id < len(self.state.visited_targets) - 1:
                        visited_targets_up_to_backtrack = self.state.visited_targets[:target_waypoint_id + 1]
                        failed_path = self.state.visited_targets[target_waypoint_id + 1:]
                        backtrack_point = visited_targets_up_to_backtrack[-1]
                        if self.reasoning.backtrack_second_chance:
                            self.logger.info(f'Initiating Backtrack Re-planning to Waypoint {target_waypoint_id}')
                            new_direction, replan_response = self.reasoning.replan_at_backtrack(
                                self.instruction, visited_targets_up_to_backtrack, failed_path,
                                episode_id=self.current_episode_id, step=runtime.step,
                                history_images=self.state.history_images if self.use_continuous_history else None)
                            self.latest_la_output = replan_response
                            self.logger.info(f'Re-planned direction: {new_direction}')
                            panorama_frames = backtrack_point['panorama_frames']
                            direction_map = {'forward': 0, 'left': 90, 'behind': 180, 'right': 270}
                            target_angle = direction_map.get(new_direction, 0)
                            frame_idx = target_angle // 90
                            if frame_idx < len(panorama_frames):
                                dir_rgb = panorama_frames[frame_idx]['rgb']
                                dir_depth = panorama_frames[frame_idx]['depth']
                                bbox = self.reasoning.query_llm(
                                    instruction=self.instruction, visited_targets=visited_targets_up_to_backtrack,
                                    rgb_image=dir_rgb, depth_image=dir_depth,
                                    width=view.width, height=view.height, current_step=runtime.current_step,
                                    progress_analysis='Backtracking and Replanning', episode_id=self.current_episode_id)
                                captured = backtrack_point['_views'][new_direction if new_direction in ('forward', 'left', 'behind', 'right') else 'forward']
                                fallback = runtime.map_goal(*backtrack_point['world_coords'])
                                goal = self._project(runtime, bbox, captured, fallback=fallback)
                                (self.target_map_x, self.target_map_y) = (goal.x, goal.y)
                            revisit_waypoint = {
                                'step': runtime.step, 'init_image': backtrack_point['init_image'],
                                'panorama_frames': backtrack_point['panorama_frames'],
                                'world_coords': backtrack_point['world_coords'], 'full_pose': backtrack_point['full_pose'],
                                'is_revisit': True, 'original_waypoint_id': target_waypoint_id,
                                'backtrack_from_waypoint_id': len(self.state.visited_targets) - 1,
                                '_views': backtrack_point['_views'],
                            }
                            self.state.visited_targets.append(revisit_waypoint)
                            self.state.total_waypoints += 1
                            self.state.visited_targets[-1]['direction_decision'] = new_direction
                            self.state.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb, np.ndarray) else dir_rgb
                            self.state.visited_targets[-1]['bbox'] = bbox
                            self.state.visited_targets[-1]['progress_analysis'] = 'Backtracking Re-plan'
                            self.is_backtracking = False
                            self.backtracking_only = False
                        else:
                            self.logger.info(f'Initiating Backtrack-only return to Waypoint {target_waypoint_id}')
                            if backtrack_point.get('world_coords') is not None:
                                (self.target_map_x, self.target_map_y) = backtrack_point['world_coords']
                            else:
                                decision['action'] = 'NAVIGATE'
                                self.panorama_got = True
                                return False
                            revisit_waypoint = {
                                'step': runtime.step, 'init_image': backtrack_point['init_image'],
                                'panorama_frames': backtrack_point['panorama_frames'],
                                'world_coords': backtrack_point['world_coords'], 'full_pose': backtrack_point['full_pose'],
                                'is_revisit': True, 'original_waypoint_id': target_waypoint_id,
                                'backtrack_from_waypoint_id': len(self.state.visited_targets) - 1,
                                'progress_analysis': 'Backtracking Only', '_views': backtrack_point['_views'],
                            }
                            self.state.visited_targets.append(revisit_waypoint)
                            self.state.total_waypoints += 1
                            self.is_backtracking = True
                            self.backtracking_only = True
                        self.target_set_step = runtime.step
                    else:
                        decision['action'] = 'NAVIGATE'
                    self.panorama_got = True
                if decision.get('action', 'NAVIGATE') == 'NAVIGATE':
                    self.navigate_or_not = True
                    direction = decision.get('direction', 'forward')
                    if decision.get('stop_signal', False):
                        self.current_la_action = 'STOP'
                    else:
                        self.current_la_action = 'NAVIGATE'
                    progress_analysis = decision.get('progress_analysis', '')
                    reasoning = decision.get('reasoning', '')
                    self.state.visited_targets[-1].update({'progress_analysis': progress_analysis, 'reasoning': reasoning, 'direction_decision': direction})
                    direction_map = {'forward': 0, 'left': 90, 'behind': 180, 'right': 270}
                    target_angle = direction_map.get(direction, 0)
                    frame_idx = target_angle // 90
                    if frame_idx < len(panorama_frames):
                        dir_rgb = panorama_frames[frame_idx]['rgb']
                        self.state.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb, np.ndarray) else dir_rgb
                        self.state.visited_targets[-1]['turn_action'] = f'turn {direction}'
                    if direction == 'left':
                        runtime.turn(90)
                    elif direction == 'right':
                        runtime.turn(-90)
                    elif direction == 'behind':
                        runtime.turn(180)
                    self.panorama_got = True
            elif self.target_map_x is None and self.panorama_got and (not runtime.busy):
                progress_analysis = self.state.visited_targets[-1].get('progress_analysis', '')
                bbox = self.reasoning.query_llm(
                    instruction=self.instruction, visited_targets=self.state.visited_targets,
                    rgb_image=view.image, depth_image=view.depth, width=view.width, height=view.height,
                    current_step=runtime.current_step, progress_analysis=progress_analysis,
                    planned_action=self.current_la_action, episode_id=self.current_episode_id)
                self.state.visited_targets[-1].update({'description': bbox.get('target', 'unknown target'), 'bbox': bbox, 'llm_reasoning': bbox.get('reasoning', ''), 'llm_progress': bbox.get('progress', '')})
                if self.current_la_action == 'STOP':
                    self.going_to_stop = True
                goal = self._project(runtime, bbox, view, adjust_depth=True)
                (self.target_map_x, self.target_map_y) = (goal.x, goal.y)
                self.target_set_step = runtime.step
                runtime.queue_navigation(goal, initial=True)
                self.panorama_got = False
            elif self.target_map_x is not None and self.target_map_y is not None and (not runtime.busy):
                runtime.queue_navigation(self.target(runtime))

        runtime.terrain_hint = self.reasoning.stair
        if runtime.busy:
            progress('executing')
            self.last_execution = runtime.execute_pending()
        self.state.step = runtime.step
        self.state.todo_list = self.reasoning.todo_list
        if self._eqa_done_requested or not runtime.running:
            self.reason = 'model_stop' if self.stop_requested else 'environment_finished'
            return True
        return False

    def finish(self, progress):
        # The old evaluator's EQA metric performed this fallback, including on
        # natural environment termination. Keep the answer in the task result.
        if (self.task.kind == 'eqa' and self.task.answer_from_history
                and self.last_submitted_answer is None and self.state.visited_targets):
            frames = self.state.visited_targets[-1].get('panorama_frames', [])
            if frames:
                progress('answering')
                try:
                    self.last_submitted_answer = self.reasoning.query_llm_oracle(frames, self.instruction)
                except Exception as error:
                    self.answer_error = str(error)
                    self.logger.error(f"EQA forced-answer failed: {error}")
