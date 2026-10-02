"""LA/VA reasoning agent (VLMReasoningAgent).

Split out of ZS_Evaluator_mp.py: this module holds the language/vision reasoning
agent (prompt construction, navigate/backtrack, TODO-list memory, STOP double-check,
replanning). The evaluator in ZS_Evaluator_mp.py instantiates it per episode.
"""
import base64
import io
import json
import os
import re
from typing import List, Dict

import numpy as np
from PIL import Image

import logging

from .habitat_prompts import *  # prompt constants used throughout the agent
from .habitat_prompts import get_prompts

class VLMReasoningAgent:
    """Simulator reasoning profile, with model and recording supplied by the host."""

    def __init__(
        self,
        model,
        image_height,
        task_type: str = "VLN",
        use_guideline=True,
        use_working_memory=True,
        allow_move_behind=True,
        debug_logging=False,
        log_dir="logs/debug_logs",
        use_todo_list=True,
        backtrack_second_chance=True,
        use_backtrack=True,
        on_bbox=None,
        log=None,
    ):
        self.logger = log or logging.getLogger(__name__)
        self.model = model
        self.model.eval()
        self.image_height = image_height
        self.on_bbox = on_bbox
        self.stair = False
        self.use_guideline = False
        self.use_working_memory = use_working_memory
        self.use_negative_constraints = True  # Switch to control negative constraints
        self.allow_move_behind = allow_move_behind
        self.use_todo_list = use_todo_list
        self.backtrack_second_chance = backtrack_second_chance
        self.task_type = task_type
        self.P = get_prompts(task_type)
        self.guideline = ""
        self.todo_list = None
        self.todo_verification_feedback = ""
        self.debug_logging = debug_logging
        self.log_dir = log_dir
        self.last_object = ''
        self.last_va_log_dir = None  # cached path of most recent VA save dir

    def _ensure_va_in_step(self, episode_id, step):
        """Ensure step_<step>/ has a VA save. If VA wasn't called this step, copy
        the last VA save dir as `va_query_cached_<ts>/`."""
        if not self.debug_logging or episode_id is None or step is None:
            return
        if not self.last_va_log_dir or not os.path.isdir(self.last_va_log_dir):
            return
        try:
            import glob, shutil, time
            step_dir = os.path.join(self.log_dir, str(episode_id), f"step_{step}")
            existing = glob.glob(os.path.join(step_dir, 'va_query_*'))
            if existing:
                return
            os.makedirs(step_dir, exist_ok=True)
            dst = os.path.join(step_dir, f"va_query_cached_{time.time()}")
            shutil.copytree(self.last_va_log_dir, dst)
        except Exception as e:
            self.logger.warning(f'failed to mirror VA into step_{step}: {e}')

    def _ensure_last_object(self, instruction: str, initial_views: List[Dict] = None,
                             episode_id=None, step=None):
        if self.last_object:
            return
        # Extracting a "last object" from a free-text instruction is VLN-specific;
        # for ObjectNav and EQA the target/question is already structured, so skip.
        if self.task_type != "VLN":
            self.last_object = ''
            return

        prompt_text = self.P.LA_PROMPT_LAST_OBJECT_GENERATOR.format(instruction=instruction)
        self.logger.info("Generating Last Object...")
        self.logger.info(prompt_text)

        content = [{"type": "text", "text": prompt_text}]
        if initial_views:
            content.extend(initial_views)

        messages = [{"role": "user", "content": content}]

        log_path = None
        if self.debug_logging and episode_id is not None:
            import time
            log_path = os.path.join(self.log_dir, str(episode_id),
                                    f"step_{step if step is not None else 0}",
                                    f"la_last_object_{time.time()}")

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=4096,
            temperature=0.7,
            use_la=True,
            log_path=log_path,
        )

        self.logger.info('Last Object:')
        self.logger.info("%s", output_text)
        self.last_object = output_text.strip()

    def _build_nav_prompt(self, use_backtrack_prompt, negative_constraints, action_list_items, action_desc, feedback):
        prompt_kwargs = {
            "negative_constraints": negative_constraints,
            "action_list": "\n".join(action_list_items),
            "action_desc": action_desc,
            "feedback": feedback,
        }

        if self.use_todo_list:
            prompt_kwargs["todo_list"] = self._format_todo_for_prompt()
        else:
            prompt_kwargs["todo_list"] = "(none)"
        template = self.P.LA_PROMPT_BACKTRACK if use_backtrack_prompt else self.P.LA_PROMPT_NO_BACKTRACK

        return template.format(**prompt_kwargs)

    def img_to_base64(self, img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64

    def _get_initial_views(self, visited_targets):
        """Helper to extract initial views from visited_targets."""
        initial_views = []
        if visited_targets and len(visited_targets) > 0:
            panorama_images = visited_targets[0]['panorama_frames']
            view_definitions = [
                {'angle': 0, 'name': 'forward', 'label': 'Initial FORWARD view'},
                {'angle': 90, 'name': 'left', 'label': 'Initial LEFT view'},
                {'angle': 180, 'name': 'behind', 'label': 'Initial BEHIND view'},
                {'angle': 270, 'name': 'right', 'label': 'Initial RIGHT view'}
            ]
            for view in view_definitions:
                angle = view['angle']
                frame_idx = angle // 90
                if frame_idx < len(panorama_images):
                    rgb_image = panorama_images[frame_idx]['rgb']
                    if isinstance(rgb_image, np.ndarray):
                        if rgb_image.dtype != np.uint8:
                            rgb_image = (rgb_image * 255).astype(np.uint8)
                        img = Image.fromarray(rgb_image)
                    else:
                        img = rgb_image
                    initial_views.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                    initial_views.append({"type": "text", "text": view['label']})
        return initial_views

    def _check_blocked_directions(self, panorama_images):
        """
        Helper to check blocked directions based on depth.
        Returns a set of blocked direction names.
        """
        blocked_directions = set()
        num_frames = len(panorama_images)
        
        view_definitions = [
            {'angle': 0, 'name': 'forward'},
            {'angle': 90, 'name': 'left'},
            {'angle': 180, 'name': 'behind'},
            {'angle': 270, 'name': 'right'}
        ]
        
        for view in view_definitions:
            angle = view['angle']
            if num_frames > 0:
                frame_idx = int((angle / 360.0) * num_frames) % num_frames
            else:
                frame_idx = 0

            if frame_idx < len(panorama_images):
                # Check depth for blockage
                if 'depth' in panorama_images[frame_idx]:
                    depth_image = panorama_images[frame_idx]['depth']
                    # Handle potential (H, W, 1) shape
                    if len(depth_image.shape) == 3:
                        d_img = depth_image[:, :, 0]
                    else:
                        d_img = depth_image
                    
                    # Check center region (middle 1/3)
                    h, w = d_img.shape
                    center_d = d_img[h//3:2*h//3, w//3:2*w//3]
                    
                    # Assuming depth is normalized [0, 1] where small value means close obstacle
                    # Filter out 0.0 if it represents invalid/far, but keep small positive values
                    valid_mask = center_d > 0.01
                    if np.any(valid_mask):
                        mean_d = np.mean(center_d[valid_mask])
                        # Threshold 0.15 corresponds to roughly 0.8m - 1.5m depending on max_depth
                        if mean_d < 0.15: 
                            blocked_directions.add(view['name'])
                            
        return blocked_directions

    def navigate_or_backtrack(self, instruction, visited_targets, feedback="", episode_id=None, step=None, history_images=None):
        """Decide the next language-level action from instruction + history + 4-dir images.

        Returns one of: `navigate to {left,right,forward,behind}` or
        `backtrack to <waypoint_id>`.
        """

        panorama_images = visited_targets[-1]['panorama_frames'] if visited_targets else []

        history_content = []
        
        if history_images is not None:
            current_waypoint_idx = 0
            
            # Create a map of step -> waypoint_id
            step_to_waypoint = {}
            
            sorted_waypoints = sorted(visited_targets, key=lambda x: x['step'])
            
            for i, target in enumerate(sorted_waypoints):
                step_to_waypoint[target['step']] = i
            
            last_wp_idx = -1
            
            for item in history_images:
                img_step = item['step']
                
                # Find the most recent waypoint index <= img_step
                current_wp_segment_idx = -1
                for wp_step, wp_idx in step_to_waypoint.items():
                    if wp_step <= img_step:
                        if wp_idx > current_wp_segment_idx:
                            current_wp_segment_idx = wp_idx
                
                # If we moved to a new segment, insert header
                if current_wp_segment_idx > last_wp_idx:
                    leg_idx = -1
                    for i in range(len(sorted_waypoints) - 1):
                        wp_curr = sorted_waypoints[i]
                        wp_next = sorted_waypoints[i+1]
                        if wp_curr['step'] <= img_step < wp_next['step']:
                            leg_idx = i
                            break
                    
                    # Handle last segment (after last completed waypoint, before current)
                    if leg_idx == -1:
                        if len(sorted_waypoints) > 0 and img_step >= sorted_waypoints[-1]['step']:
                            leg_idx = len(sorted_waypoints) - 1
 

            last_leg_idx = -2
            
            for item in history_images:
                img_step = item['step']
                img = item['image']
                
                leg_idx = -1
                
                # Find interval [WP_i, WP_i+1)
                for i in range(len(sorted_waypoints) - 1):
                    if sorted_waypoints[i]['step'] <= img_step < sorted_waypoints[i+1]['step']:
                        leg_idx = i
                        break
                
                # If not found in intervals, check if it's after the last waypoint
                if leg_idx == -1 and len(sorted_waypoints) > 0:
                    if img_step >= sorted_waypoints[-1]['step']:
                        leg_idx = len(sorted_waypoints) - 1
                        
                # If leg changed, insert header
                if leg_idx != last_leg_idx:
                    if leg_idx >= 0:
                        # Determine label

                        start_wp = sorted_waypoints[leg_idx]
                        start_id = leg_idx
                        
                        if leg_idx < len(sorted_waypoints) - 1:
                            end_id = leg_idx + 1
                            label = f"Waypoint {start_id} -> Waypoint {end_id}: "
                        else:
                            # Last segment
                            label = f"Waypoint {start_id} -> Current Position: "
                            
                        history_content.append({"type": "text", "text": label})
                    last_leg_idx = leg_idx

                history_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
        else:
            # Add history waypoint images and action descriptions
            for i, target in enumerate(visited_targets[:-1]):  # exclude the in-progress waypoint
                if 'init_image' in target:
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: Arrival view"})

                if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: After turn {turn_direction} view"})

                if 'description' in target:
                    history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

        current_views = []
        view_definitions = [
            {'angle': 0, 'name': 'forward', 'label': 'Current FORWARD view:'},
            {'angle': 90, 'name': 'left', 'label': 'Current LEFT view:'},
            {'angle': 180, 'name': 'behind', 'label': 'Current BEHIND view:'},
            {'angle': 270, 'name': 'right', 'label': 'Current RIGHT view:'}
        ]

        blocked_directions = self._check_blocked_directions(panorama_images)
        num_frames = len(panorama_images)
        
        for view in view_definitions:
            angle = view['angle']
            # Calculate frame index based on total frames (assuming uniform distribution)
            # For 12 frames: 0->0, 90->3, 180->6, 270->9
            if num_frames > 0:
                frame_idx = int((angle / 360.0) * num_frames) % num_frames
            else:
                frame_idx = 0

            if frame_idx < len(panorama_images):
                rgb_image = panorama_images[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                current_views.append({"type": "text", "text": view['label']})
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})

        # Always offer the backtrack action when there are prior waypoints;
        # filtering by distance happens below in the available_ids block.
        num_waypoints = len([t for t in visited_targets[:-1] if 'description' in t])
        should_consider_backtrack = 1

        content = [{"type": "text", "text": f"Navigation Task: \"{instruction}\"\n\nNavigation History:"}]
        content.extend(history_content)
        if self.use_guideline and hasattr(self, 'guideline') and self.guideline:
            pass
            # content.append({"type": "text", "text": "\nTask-specific guidelines:"})
            # content.append({"type": "text", "text": self.guideline})
        # content.append()
        if self.use_working_memory and len(visited_targets) >= 2:
            # Working memory
            content.append({"type": "text", "text": "\nWorking Memory (previous waypoint snapshot): compare these with current views to judge progress, repeated scenes."})
            target = visited_targets[-2]
            view_definitions = [
                {'angle': 0, 'name': 'forward', 'label': 'Previous FORWARD view'},
                {'angle': 90, 'name': 'left', 'label': 'Previous LEFT view'},
                {'angle': 180, 'name': 'behind', 'label': 'Previous BEHIND view'},
                {'angle': 270, 'name': 'right', 'label': 'Previous RIGHT view'}
            ]
            previous_views = []

            for view in view_definitions:
                angle = view['angle']
                frame_idx = angle // 90
                if frame_idx < len(panorama_images):
                    rgb_image = panorama_images[frame_idx]['rgb']
                    if isinstance(rgb_image, np.ndarray):
                        if rgb_image.dtype != np.uint8:
                            rgb_image = (rgb_image * 255).astype(np.uint8)
                        img = Image.fromarray(rgb_image)
                    else:
                        img = rgb_image
                    previous_views.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                    previous_views.append({"type": "text", "text": view['label']})
                
            content.extend(previous_views)   


        content.append({"type": "text", "text": "\nCurrent 4-directional views:"})
        content.extend(current_views)

        # Ensure reasoning state is available for the current episode.
        if self.use_todo_list and self.todo_list is None:
            initial_views = self._get_initial_views(visited_targets)
            self.generate_todo_list(instruction, initial_views, episode_id=episode_id, step=step)
        elif not self.last_object:
            initial_views = self._get_initial_views(visited_targets)
            self._ensure_last_object(instruction, initial_views, episode_id=episode_id, step=step)

        # Generate negative constraints based on target object
        target_objects = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
        target = None
        instruction_lower = instruction.lower()
        for obj in target_objects:
            obj_name = obj.replace('_', ' ')
            if obj_name in instruction_lower or obj in instruction_lower:
                target = obj
                break
        
        negative_constraints = ""
        if self.use_negative_constraints:
            if target:
                other_objects = [obj.replace('_', ' ') for obj in target_objects if obj != target]
                negative_constraints = f"- When looking for {target.replace('_', ' ')}, do not confuse it with other objects: {', '.join(other_objects)}."

        # Generate action list dynamically
        action_list_items = []
        action_desc_parts = []
        
        # Directions we want to consider
        candidate_directions = ['forward', 'left', 'right']
        
        # Check if we are at the initial position (only current waypoint exists)
        is_initial_position = len(visited_targets) <= 1
        if self.allow_move_behind or is_initial_position:
            candidate_directions.append('behind')

        # Build action list based on valid directions (using blocked_directions calculated earlier)
        valid_directions = [d for d in candidate_directions if d not in blocked_directions]
        
        # Fallback: if all valid directions are blocked, allow behind or forward to avoid getting stuck
        if not valid_directions:
            if 'behind' in candidate_directions:
                valid_directions.append('behind')
            elif 'forward' in candidate_directions:
                valid_directions.append('forward')

        for direction in valid_directions:
            if direction == 'forward':
                action_list_items.append("   - navigate to forward - continue straight ahead")
            elif direction == 'left':
                action_list_items.append("   - navigate to left - turn left and go forward")
            elif direction == 'right':
                action_list_items.append("   - navigate to right - turn right and go forward")
            elif direction == 'behind':
                action_list_items.append("   - navigate to behind - turn around and go forward")
            
            action_desc_parts.append("navigate to " + direction)

        # Add STOP action - REMOVED (now handled via "stop": True key)
        # action_list_items.append("   - stop - task completed")
        # action_desc_parts.append("stop")

        if not action_desc_parts:
             # Should be covered by fallback, but just in case
             action_desc = "navigate to behind" 
             action_list_items.append("   - navigate to behind - turn around and go forward (Emergency)")
        else:
             action_desc = "|".join(action_desc_parts)



        if should_consider_backtrack and num_waypoints > 0:
            # Collect available waypoint IDs (excluding current one which is just created)
            # visited_targets includes current one at end.
            # Available indices are 0 to len(visited_targets) - 2
            
            # Filter waypoints by distance
            MAX_BACKTRACK_DISTANCE = 6.0 # meters
            current_target = visited_targets[-1]
            current_pos = current_target.get('world_coords')
            
            available_ids = []
            if current_pos:
                for i in range(len(visited_targets) - 1):
                    target = visited_targets[i]
                    target_pos = target.get('world_coords')
                    if target_pos:
                        dist_pixels = np.sqrt((current_pos[0] - target_pos[0])**2 + (current_pos[1] - target_pos[1])**2)
                        dist_meters = dist_pixels * 0.05 
                        
                        if dist_meters <= MAX_BACKTRACK_DISTANCE:
                            available_ids.append(str(i))
            else:
                # Fallback if no coords
                available_ids = [str(i) for i in range(len(visited_targets) - 1)]

            if available_ids:
                id_list_str = ", ".join(available_ids)
                action_list_items.append(f"   - backtrack to <waypoint_id> - return to a previous waypoint (Available IDs: {id_list_str})")
                action_desc += ' or "backtrack to <waypoint_id>"'
                prompt = self._build_nav_prompt(
                    use_backtrack_prompt=True,
                    negative_constraints=negative_constraints,
                    action_list_items=action_list_items,
                    action_desc=action_desc,
                    feedback=feedback,
                )
            else:
                prompt = self._build_nav_prompt(
                    use_backtrack_prompt=False,
                    negative_constraints=negative_constraints,
                    action_list_items=action_list_items,
                    action_desc=action_desc,
                    feedback=feedback,
                )
        else:
            prompt = self._build_nav_prompt(
                use_backtrack_prompt=False,
                negative_constraints=negative_constraints,
                action_list_items=action_list_items,
                action_desc=action_desc,
                feedback=feedback,
            )

        self.logger.info(prompt)

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        log_path = None
        if self.debug_logging and episode_id is not None and step is not None:
             import time
             log_path = os.path.join(self.log_dir, str(episode_id), f"step_{step}", f"la_nav_decision_{time.time()}")
             # mirror most recent VA into this step folder if VA wasn't called here
             self._ensure_va_in_step(episode_id, step)

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=8192,
            temperature=0.7,
            use_la=True,
            log_path=log_path,
        )

        self.logger.info('LA-response:')
        self.logger.info(f"{output_text}")
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)

        retry_count = 0
        max_json_retries = 5
        while not json_match and retry_count < max_json_retries:
            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=8192,
                temperature=0.7,
                use_la=True,
                log_path=log_path,
            )
            self.logger.info('Retried.')
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            retry_count += 1

        if retry_count >= max_json_retries and not json_match:
             self.logger.error("Failed to get valid JSON response after max retries")
             return {
                'action': 'NAVIGATE',
                'direction': 'forward',
                'progress_analysis': 'Failed to parse JSON',
                'reasoning': 'Max retries reached'
             }

        if json_match:
            try:
                response_data = json.loads(json_match.group())
            except:
                response_data = {}
            
            if self.use_todo_list:
                # New diff-based mechanism: apply todo_updates if present
                if 'todo_updates' in response_data and response_data['todo_updates']:
                    self._apply_todo_updates(response_data['todo_updates'])
                    self.logger.info(f"Applied TODO updates. Current list: {self.todo_list}")
                # Legacy fallback: support old 'updated_todo_list' field (full markdown rewrite)
                elif 'updated_todo_list' in response_data:
                    legacy = response_data['updated_todo_list']
                    if isinstance(legacy, list):
                        legacy = "\n".join(legacy)
                    if isinstance(self.todo_list, list) and isinstance(legacy, str):
                        # Old format received but we have new structured - ignore
                        self.logger.info("Ignoring legacy updated_todo_list field (using structured format)")
                    else:
                        self.todo_list = legacy
                        self.logger.info(f"Updated TODO List (legacy): {legacy}")

            action = response_data.get('action', 'navigate to forward')
            progress_analysis = response_data.get('progress_analysis', '')
            # Two-stage reasoning fields: reasoning_todo (for todo updates) and
            # reasoning_action (for the chosen action). Fall back to legacy 'reasoning'
            # if either is missing.
            legacy_reasoning = response_data.get('reasoning', '')
            self.reasoning_todo = response_data.get('reasoning_todo', legacy_reasoning)
            reasoning = response_data.get('reasoning_action', legacy_reasoning)
            self.stair = response_data.get('stair', False)

            if self.stair:
                self.logger.info(f"DEBUG: LA Model detected stairs: {self.stair}")
            stop_signal = response_data.get('stop', False)
            action = action.lower()

            if action == 'stop' or 'stop' in action.split():
                 stop_signal = True
                 if action == 'stop':
                     action = 'navigate to forward'

            # The LA's stop=true is trusted unconditionally; the only remaining
            # gate is the double-check STOP LLM pass below (visual verification
            # that the target is actually reached).

            if action.startswith('backtrack to'):
                waypoint_id = action.split('backtrack to ')[-1].strip()
                if waypoint_id.startswith('waypoint'):
                    waypoint_id = waypoint_id.split('waypoint')[-1].strip()
                try:
                    waypoint_id = int(waypoint_id)
                    return {
                        'action': 'BACKTRACK',
                        'waypoint': waypoint_id,
                        'progress_analysis': progress_analysis,
                        'reasoning': reasoning
                    }
                except:
                    pass  # parse failure falls through to direction parsing below

            if 'forward' in action:
                direction = 'forward'
            elif 'left' in action:
                direction = 'left'
            elif 'right' in action:
                direction = 'right'
            elif 'behind' in action:
                direction = 'behind'
            else:
                direction = 'forward'

            return {
                'action': 'NAVIGATE',
                'direction': direction,
                'progress_analysis': progress_analysis,
                'reasoning': reasoning,
                'stop_signal': stop_signal
            }

        return {
            'action': 'NAVIGATE',
            'direction': 'forward',
            'progress_analysis': 'Unable to analyze due to parsing error',
            'reasoning': 'Fallback to forward navigation'
        }

    def generate_todo_list(self, instruction: str, initial_views: List[Dict] = None,
                           episode_id=None, step=None):
        """
        Generate initial TODO list as a structured list of dicts.
        Each item: {"content": str, "status": "pending"|"in_progress"|"completed", "result": str}
        """
        if not self.use_todo_list:
            self.todo_list = None
            self._ensure_last_object(instruction, initial_views, episode_id=episode_id, step=step)
            return

        # For VLN, merge TODO generation and last-object extraction into one LA call.
        # For other task types, last_object is not used (see _ensure_last_object).
        if self.task_type == "VLN" and hasattr(self.P, "LA_PROMPT_TODO_AND_LAST_OBJECT"):
            prompt_text = self.P.LA_PROMPT_TODO_AND_LAST_OBJECT.format(instruction=instruction)
            self.logger.info("Generating initial TODO list + last object (merged)...")
            log_purpose = "la_todo_and_last_object"
        else:
            prompt_text = self.P.LA_PROMPT_TODO_GENERATOR.format(instruction=instruction)
            self.logger.info("Generating initial TODO list...")
            log_purpose = "la_todo_generator"
        self.logger.info(prompt_text)

        content = [{"type": "text", "text": prompt_text}]
        if initial_views:
            content.extend(initial_views)

        messages = [{"role": "user", "content": content}]

        log_path = None
        if self.debug_logging and episode_id is not None:
            import time
            log_path = os.path.join(self.log_dir, str(episode_id),
                                    f"step_{step if step is not None else 0}",
                                    f"{log_purpose}_{time.time()}")

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=4096,
            temperature=0.7,
            use_la=True,
            log_path=log_path,
        )

        self.logger.info('TODO+LastObject Output:' if self.task_type == "VLN" else 'TODO List Output:')
        self.logger.info("%s", output_text)

        # Parse JSON TODO list (also picks up last_object if present)
        self.todo_list = self._parse_todo_json(output_text, instruction)
        self.logger.info(f"Parsed TODO list ({len(self.todo_list)} items): {self.todo_list}")

        if self.task_type == "VLN":
            import re as _re, json as _json
            m = _re.search(r'\{[\s\S]*\}', output_text)
            if m:
                try:
                    data = _json.loads(m.group())
                    lo = data.get("last_object", "")
                    if isinstance(lo, str) and lo.strip():
                        self.last_object = lo.strip()
                        self.logger.info(f"Last Object (merged): {self.last_object}")
                except Exception as e:
                    self.logger.warning(f"Failed to parse last_object from merged output: {e}")

        # Fallback: if last_object still empty (parse failed or non-VLN), call separate API
        self._ensure_last_object(instruction, initial_views, episode_id=episode_id, step=step)

    def _parse_todo_json(self, text: str, instruction: str) -> List[Dict]:
        """Parse JSON TODO list from LA output. Fallback to single-item list if parse fails."""
        import json as _json
        import re as _re
        m = _re.search(r'\{[\s\S]*\}', text)
        if not m:
            self.logger.warning(f"Failed to parse TODO JSON, using fallback. Raw: {text[:200]}")
            return [{"content": f"Find {instruction}", "status": "pending", "result": ""}]
        try:
            data = _json.loads(m.group())
            todos = data.get("todos", [])
            if not isinstance(todos, list) or not todos:
                return [{"content": f"Find {instruction}", "status": "pending", "result": ""}]
            parsed = []
            for t in todos:
                if isinstance(t, dict) and "content" in t:
                    parsed.append({
                        "content": str(t["content"]),
                        "status": t.get("status", "pending") if t.get("status") in ("pending", "completed") else "pending",
                        "result": str(t.get("result", "")),
                    })
            if not parsed:
                return [{"content": f"Find {instruction}", "status": "pending", "result": ""}]
            return parsed
        except Exception as e:
            self.logger.warning(f"TODO JSON parse error: {e}. Raw: {text[:200]}")
            return [{"content": f"Find {instruction}", "status": "pending", "result": ""}]

    def _format_todo_for_prompt(self) -> str:
        """Render structured TODO as a compact text block for prompts."""
        if not self.todo_list:
            return "(empty)"
        if isinstance(self.todo_list, str):
            # Legacy format: already a string
            return self.todo_list
        lines = []
        for i, t in enumerate(self.todo_list):
            status = t.get("status", "pending")
            content = t.get("content", "")
            result = t.get("result", "")
            if status == "completed" and result:
                lines.append(f"[{i}] ({status})  {content}  => {result}")
            else:
                lines.append(f"[{i}] ({status})  {content}")
        return "\n".join(lines)

    def _apply_todo_updates(self, updates) -> None:
        """Apply incremental updates (diff) to self.todo_list. Silently skip invalid ops."""
        if not updates or not isinstance(updates, list):
            return
        if not isinstance(self.todo_list, list):
            # Legacy string TODO, skip updates
            return
        # Only two statuses — "pending" and "completed". No in_progress.
        VALID_STATUS = ("pending", "completed")
        for u in updates:
            if not isinstance(u, dict):
                continue
            op = u.get("op", "update")
            try:
                if op == "update":
                    idx = u.get("index")
                    if not isinstance(idx, int) or idx < 0 or idx >= len(self.todo_list):
                        continue
                    new_status = u.get("status")
                    new_result = u.get("result")
                    if new_status in VALID_STATUS:
                        # Discipline: marking "completed" without a result is rolled
                        # back to "pending" — force LA to back up the claim with an
                        # observation, or it won't count as done.
                        if new_status == "completed" and (new_result is None or not str(new_result).strip()):
                            self.logger.info(f"TODO idx={idx} completion lacks result; rolled back to pending")
                            new_status = "pending"
                        self.todo_list[idx]["status"] = new_status
                    if new_result is not None:
                        self.todo_list[idx]["result"] = str(new_result)
                elif op == "rewrite":
                    # Allow LA to refine a pending item's content. Blocked on completed.
                    idx = u.get("index")
                    if not isinstance(idx, int) or idx < 0 or idx >= len(self.todo_list):
                        continue
                    if self.todo_list[idx].get("status") == "completed":
                        self.logger.info(f"TODO idx={idx} rewrite skipped (already completed)")
                        continue
                    if "content" in u and str(u["content"]).strip():
                        self.todo_list[idx]["content"] = str(u["content"])
                    if "status" in u and u["status"] == "pending":
                        self.todo_list[idx]["status"] = "pending"  # no-op but explicit
                    if "result" in u:
                        self.todo_list[idx]["result"] = str(u["result"])
                elif op in ("add", "insert"):
                    # Unified: append by default, or insert at `index` if provided.
                    if "content" not in u:
                        continue
                    new_item = {
                        "content": str(u["content"]),
                        "status": u.get("status") if u.get("status") in VALID_STATUS else "pending",
                        "result": str(u.get("result", "")),
                    }
                    idx = u.get("index")
                    if isinstance(idx, int):
                        idx = max(0, min(idx, len(self.todo_list)))
                        self.todo_list.insert(idx, new_item)
                    else:
                        self.todo_list.append(new_item)
                elif op == "remove":
                    idx = u.get("index")
                    if isinstance(idx, int) and 0 <= idx < len(self.todo_list):
                        self.todo_list.pop(idx)
            except Exception as e:
                self.logger.warning(f"apply_todo_update skipped item {u}: {e}")


    def query_llm(self,
                  instruction: str,
                  visited_targets: List[Dict[str, str]],
                  rgb_image: np.ndarray,
                  depth_image: np.ndarray,
                  width: int,
                  height: int,
                  current_step: int,
                  progress_analysis: str = None,
                  planned_action: str = 'NAVIGATE',
                  episode_id=None):
        try:
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                img = Image.fromarray(rgb_image)
            else:
                img = rgb_image

            if isinstance(depth_image, np.ndarray):
                if depth_image.dtype != np.uint8:
                    depth_image = (depth_image * 255).astype(np.uint8)
                if depth_image.ndim == 3 and depth_image.shape[2] == 1:
                    depth_image = np.squeeze(depth_image, axis=2)
                depth_img = Image.fromarray(depth_image)
            else:
                depth_img = depth_image

            content = [{
                'type': 'text',
                'text': self.P.ROBOT_NAVIGATION_SYSTEM_PROMPT,
            }]

            img_base64 = self.img_to_base64(img)
            depth_base64 = self.img_to_base64(depth_img)

            content.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{img_base64}"}})
            content.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{depth_base64}"}})

            progress_info = ""
            if progress_analysis:
                progress_info = f"\nProgress Analysis from Navigation Decision: {progress_analysis}\n"
            
            # Re-generate negative constraints logic as it is local variable in navigate_or_backtrack
            # We need to extract it again or pass it. 
            # Since this method is called independently, we re-extract target.
            target_objects = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
            target = None
            instruction_lower = instruction.lower()
            for obj in target_objects:
                obj_name = obj.replace('_', ' ')
                if obj_name in instruction_lower or obj in instruction_lower:
                    target = obj
                    break
            
            negative_constraints = ""
            if self.use_negative_constraints:
                if target:
                    other_objects = [obj.replace('_', ' ') for obj in target_objects if obj != target]
                    negative_constraints = f"- When looking for {target.replace('_', ' ')}, do not confuse it with other objects: {', '.join(other_objects)}."

            if self.use_todo_list:
                if self.todo_list is None:
                    todo_text = "No TODO list available."
                else:
                    todo_text = self._format_todo_for_prompt()
            else:
                todo_text = "(none)"
            prompt = self.P.VA_PROMPT.format(
                instruction=instruction,
                current_step=current_step,
                width=width,
                height=height,
                todo_list=todo_text,
                progress_info=progress_info,
                planned_action=planned_action,
            )
            self.logger.info(prompt)
            content.append({
                "type": "text",
                "text": prompt
            })

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            log_path = None
            if self.debug_logging and episode_id is not None:
                import time, glob, shutil
                step_dir = os.path.join(self.log_dir, str(episode_id), f"step_{current_step}")
                # remove any prior cached VA mirror in this step — a real VA call now happens
                for _stale in glob.glob(os.path.join(step_dir, 'va_query_cached_*')):
                    try:
                        shutil.rmtree(_stale)
                    except Exception:
                        pass
                log_path = os.path.join(step_dir, f"va_query_{time.time()}")
                self.last_va_log_dir = log_path

            # Use the API client to generate response
            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=10240,
                temperature=0,
                log_path=log_path,
                extra_body={"enable_thinking": False}
            )
            self.logger.info('LLM Output:')
            self.logger.info("%s", output_text)

            # Try to parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            retry_count = 0
            max_json_retries = 5
            while not json_match and retry_count < max_json_retries:
                output_text = self.model.generate(
                    messages=messages,
                    max_new_tokens=4096,
                    temperature=0.0,
                    log_path=log_path,
                    extra_body={"enable_thinking": False}
                )
                self.logger.info('Retried LLM Output:')
                self.logger.info("%s", output_text)
                json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
                retry_count += 1

            if retry_count >= max_json_retries and not json_match:
                 self.logger.error("Failed to get valid JSON response in query_llm after max retries")
                 # Fallback will be handled by exception or check below
                 json_match = None

            if json_match:
                try:
                    response_data = json.loads(json_match.group())
                except:
                    response_data = {}

                # Extract action decision
                action_decision = response_data.get('action', planned_action).upper()

                # Extract bbox_2d in [x1, y1, x2, y2] format
                bbox_2d = response_data.get('bbox_2d', [width // 4, height // 4, 3 * width // 4,
                                                        3 * height // 4])

                # Ensure we have 4 coordinates
                if len(bbox_2d) >= 4:
                    x1, y1, x2, y2 = bbox_2d[:4]
                    x1, y1, x2, y2 = x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height
                else:
                    # Fallback if bbox_2d is malformed
                    x1, y1, x2, y2 = width // 4, height // 4, 3 * width // 4, 3 * height // 4

                # Convert to x, y, width, height format for internal use
                x = int(x1)
                y = int(y1)
                img_w, img_h = width, height
                width = int(x2 - x1)
                height = int(y2 - y1)

                bbox = {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'target': response_data.get('target', 'unknown target'),
                    'action': action_decision,
                    'reasoning': response_data.get('reasoning', 'No reasoning provided'),
                    'progress': response_data.get('progress', 'Unknown progress')
                }

                # Ensure bbox is within image bounds
                bbox['x1'] = max(0, min(bbox['x1'], img_w - 1))
                bbox['y1'] = max(0, min(bbox['y1'], img_h - 1))
                bbox['x2'] = max(bbox['x1'] + 1, min(bbox['x2'], img_w))
                bbox['y2'] = max(bbox['y1'] + 1, min(bbox['y2'], img_h))

                # Update x, y, width, height based on bounded coordinates
                bbox['x'] = bbox['x1']
                bbox['y'] = bbox['y1']
                bbox['width'] = bbox['x2'] - bbox['x1']
                bbox['height'] = bbox['y2'] - bbox['y1']

                # Record this target if it's a new navigation target
                if action_decision == 'NAVIGATE' and bbox['target'] != 'unknown target':
                    target_record = {
                        'step': current_step,
                        'description': bbox['target'],
                        'bbox': {
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'x1': bbox['x1'],
                            'y1': bbox['y1'],
                            'x2': bbox['x2'],
                            'y2': bbox['y2']
                        },
                        'reasoning': bbox['reasoning']
                        # world_coords is set at waypoint creation, not here.
                    }

                    if visited_targets is not None:
                        visited_targets[-1].update(target_record)

                # Calculate target coordinates for visualization
                if not self.stair:
                    # coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int((bbox.get('y1', 0) + 3 * bbox.get('y2', 0)) / 4.0))
                    coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), min(int(bbox.get('y2', 0)), self.image_height - 1))
                else:
                    # Stair navigation
                    # If 'up' (or True for legacy), target top (y1). If 'down', target bottom (y2).
                    if self.stair == 'down':
                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y2', 0)))
                    else:
                        # Default to 'up' behavior (y1)
                        coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y1', 0)))

                # Save RGB image with bounding box annotation
                if self.on_bbox is not None:
                    self.on_bbox(rgb_image, bbox, target_coords=coords)

                return bbox
            else:
                self.logger.info("Failed to parse JSON response, using default")

        except Exception as e:
            self.logger.info(f"Error in query_llm: {e}")

        # Fallback bbox (center of image)
        x1_fallback = width // 4
        y1_fallback = height // 4
        x2_fallback = 3 * width // 4
        y2_fallback = 3 * height // 4

        fallback_bbox = {
            'x': x1_fallback,
            'y': y1_fallback,
            'width': x2_fallback - x1_fallback,
            'height': y2_fallback - y1_fallback,
            'x1': x1_fallback,
            'y1': y1_fallback,
            'x2': x2_fallback,
            'y2': y2_fallback,
            'target': 'fallback target',
            'action': 'NAVIGATE',
            'reasoning': 'Fallback due to parsing error',
            'progress': 'Unknown due to error'
        }

        # Calculate fallback coords
        if not self.stair:
            coords_fallback = (int((fallback_bbox.get('x1', 0) + fallback_bbox.get('x2', 0)) / 2.0), min(int(fallback_bbox.get('y2', 0)), self.image_height - 1))
        else:
            if self.stair == 'down':
                coords_fallback = (int((fallback_bbox.get('x1', 0) + fallback_bbox.get('x2', 0)) / 2.0), int(fallback_bbox.get('y2', 0)))
            else:
                coords_fallback = (int((fallback_bbox.get('x1', 0) + fallback_bbox.get('x2', 0)) / 2.0), int(fallback_bbox.get('y1', 0)))

        # Save RGB image with fallback bounding box
        if self.on_bbox is not None:
            self.on_bbox(rgb_image, fallback_bbox, target_coords=coords_fallback)

        return fallback_bbox

    def double_check_stop(self, instruction, panorama_frames, visited_targets, episode_id=None, step=None, history_images=None):
        """
        Double check if the agent should really stop based on 4-directional views.
        Returns: True if should stop, False otherwise.
        """
        # return True, {}
        # Construct current views
        current_views = []
        view_definitions = [
            {'angle': 0, 'name': 'forward', 'label': 'Current FORWARD view'},
            {'angle': 90, 'name': 'left', 'label': 'View after turning LEFT'},
            {'angle': 180, 'name': 'behind', 'label': 'View after turning BEHIND'},
            {'angle': 270, 'name': 'right', 'label': 'View after turning RIGHT'}
        ]

        for view in view_definitions:
            angle = view['angle']
            frame_idx = angle // 90
            if frame_idx < len(panorama_frames):
                rgb_image = panorama_frames[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                current_views.append({"type": "text", "text": view['label']})

        history_content = []
        
        if history_images is not None:
            current_waypoint_idx = 0
            
            # Create a map of step -> waypoint_id
            step_to_waypoint = {}
            
            sorted_waypoints = sorted(visited_targets, key=lambda x: x['step'])
            
            for i, target in enumerate(sorted_waypoints):
                step_to_waypoint[target['step']] = i
            
            last_wp_idx = -1
            
            for item in history_images:
                img_step = item['step']
                
                # Find the most recent waypoint index <= img_step
                current_wp_segment_idx = -1
                for wp_step, wp_idx in step_to_waypoint.items():
                    if wp_step <= img_step:
                        if wp_idx > current_wp_segment_idx:
                            current_wp_segment_idx = wp_idx
                
                # If we moved to a new segment, insert header
                if current_wp_segment_idx > last_wp_idx:
                    leg_idx = -1
                    for i in range(len(sorted_waypoints) - 1):
                        wp_curr = sorted_waypoints[i]
                        wp_next = sorted_waypoints[i+1]
                        if wp_curr['step'] <= img_step < wp_next['step']:
                            leg_idx = i
                            break
                    
                    # Handle last segment (after last completed waypoint, before current)
                    if leg_idx == -1:
                        if len(sorted_waypoints) > 0 and img_step >= sorted_waypoints[-1]['step']:
                            leg_idx = len(sorted_waypoints) - 1
 

            last_leg_idx = -2
            
            for item in history_images:
                img_step = item['step']
                img = item['image']
                
                leg_idx = -1
                
                # Find interval [WP_i, WP_i+1)
                for i in range(len(sorted_waypoints) - 1):
                    if sorted_waypoints[i]['step'] <= img_step < sorted_waypoints[i+1]['step']:
                        leg_idx = i
                        break
                
                # If not found in intervals, check if it's after the last waypoint
                if leg_idx == -1 and len(sorted_waypoints) > 0:
                    if img_step >= sorted_waypoints[-1]['step']:
                        leg_idx = len(sorted_waypoints) - 1
                        
                # If leg changed, insert header
                if leg_idx != last_leg_idx:
                    if leg_idx >= 0:
                        # Determine label

                        start_wp = sorted_waypoints[leg_idx]
                        start_id = leg_idx
                        
                        if leg_idx < len(sorted_waypoints) - 1:
                            end_id = leg_idx + 1
                            label = f"Waypoint {start_id} -> Waypoint {end_id}: "
                        else:
                            # Last segment
                            label = f"Waypoint {start_id} -> Current Position: "
                            
                        history_content.append({"type": "text", "text": label})
                    last_leg_idx = leg_idx

                history_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
        else:
            # Add history waypoint images and action descriptions
            for i, target in enumerate(visited_targets[:-1]):  # exclude the in-progress waypoint
                if 'init_image' in target:
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: Arrival view"})


                if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: After turn {turn_direction} view"})

                if 'description' in target:
                    history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})


        # Construct prompt
        parts = STOP_CHECK_PROMPT.split("{current_views}")
        part1 = parts[0].format(instruction=instruction, target=self.last_object)
        rest = parts[1]
        
        content = [{"type": "text", "text": part1}]
        content.extend(current_views)
        content.append({"type": "text", "text": rest})

        messages = [{"role": "user", "content": content}]
        
        self.logger.info("Double-checking stop decision...")
        
        log_path = None
        if self.debug_logging and episode_id is not None and step is not None:
            import time
            log_path = os.path.join(self.log_dir, str(episode_id), f"step_{step}", f"la_double_check_{time.time()}")

        # Call LLM
        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=8192,
            temperature=0.7, # Low temperature for decision
            use_la=True,  # Use gemini-2.5-pro
            log_path=log_path
        )
        
        self.logger.info('Double-check response:')
        self.logger.info(f"{output_text}")
        
        # Parse response
        import json
        import re
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        
        if json_match:
            try:
                response_data = json.loads(json_match.group())
                decision = response_data.get('decision', 'CONTINUE').upper()
                return decision == 'STOP', response_data
            except:
                self.logger.warning("Failed to parse double-check response JSON")
                return False, {}
        
        return False, {}

    def replan_at_backtrack(self, instruction, visited_targets_up_to_backtrack, failed_path, episode_id=None, step=None, history_images=None):
        """
        Re-plan navigation from a backtrack point.
        """
        # 1. Construct History (up to backtrack point)
        history_content = []
        failed_path_content = []

        if history_images is not None:
             # Logic for continuous history images
             # Determine backtrack step (step when the backtrack waypoint was created/visited)
             backtrack_target = visited_targets_up_to_backtrack[-1]
             backtrack_step = backtrack_target['step']
             
             # Group logic similar to navigate_or_backtrack
             sorted_waypoints = sorted(visited_targets_up_to_backtrack, key=lambda x: x['step'])
             last_leg_idx = -2
             
             for item in history_images:
                 img_step = item['step']
                 img = item['image']
                 
                 # Determine if this image is part of history or failed path
                 if img_step <= backtrack_step:
                     target_list = history_content
                     
                     # Add segment header for history
                     leg_idx = -1
                     for i in range(len(sorted_waypoints) - 1):
                         if sorted_waypoints[i]['step'] <= img_step < sorted_waypoints[i+1]['step']:
                             leg_idx = i
                             break
                     if leg_idx == -1 and len(sorted_waypoints) > 0:
                         if img_step >= sorted_waypoints[-1]['step']:
                             leg_idx = len(sorted_waypoints) - 1
                             
                     if leg_idx != last_leg_idx:
                         if leg_idx >= 0:
                             start_id = leg_idx
                             if leg_idx < len(sorted_waypoints) - 1:
                                 end_id = leg_idx + 1
                                 label = f"Waypoint {start_id} -> Waypoint {end_id}: "
                             else:
                                 # For history up to backtrack, the last segment ends at the backtrack waypoint
                                 label = f"Waypoint {start_id} -> Waypoint {start_id} (Backtrack Point): "
                             target_list.append({"type": "text", "text": label})
                         last_leg_idx = leg_idx
                 else:
                     target_list = failed_path_content
                     
                     if not failed_path_content: # First item
                         target_list.append({"type": "text", "text": "Trajectory after Backtrack Point (Failed Path):"})

                 img_data = {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(img)}"
                 }}
                 target_list.append(img_data)
        else:
            for i, target in enumerate(visited_targets_up_to_backtrack):
                 if 'init_image' in target:
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: Arrival view"})
                 if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    history_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    history_content.append({"type": "text", "text": f"Waypoint {i}: After turn {turn_direction} view"})
                 if 'description' in target:
                    history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

            # 2. Construct Failed Path Description (Images + Text)
            for i, target in enumerate(failed_path):
                idx = len(visited_targets_up_to_backtrack) + i
                if 'init_image' in target:
                    failed_path_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                    failed_path_content.append({"type": "text", "text": f"Waypoint {idx}: Arrival view"})

                if 'dir_image' in target:
                    turn_direction = target.get('direction_decision', '')
                    failed_path_content.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                    failed_path_content.append({"type": "text", "text": f"Waypoint {idx}: After turn {turn_direction} view"})
                if 'description' in target:
                    failed_path_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

        if not failed_path_content and not history_images:
             failed_path_content.append({"type": "text", "text": "None (Immediate failure)"})

        # 3. Construct Backtrack Point Views (Current Views) and Action List
        backtrack_point = visited_targets_up_to_backtrack[-1]
        panorama_frames = backtrack_point['panorama_frames']
        
        current_views = []
        view_definitions = [
            {'angle': 0, 'name': 'forward', 'label': 'Current FORWARD view:'},
            {'angle': 90, 'name': 'left', 'label': 'View after turning LEFT:'},
            {'angle': 180, 'name': 'behind', 'label': 'View after turning BEHIND:'},
            {'angle': 270, 'name': 'right', 'label': 'View after turning RIGHT:'}
        ]
        
        # Calculate blocked directions for backtrack point
        blocked_directions = self._check_blocked_directions(panorama_frames)
        
        # Generate Action List
        candidate_directions = ['forward', 'left', 'right', 'behind']
        valid_directions = [d for d in candidate_directions if d not in blocked_directions]
        
        # Fallback if all blocked
        if not valid_directions:
            if 'behind' in candidate_directions: valid_directions.append('behind')
            elif 'forward' in candidate_directions: valid_directions.append('forward')

        action_list_items = []
        action_desc_parts = []
        
        for direction in valid_directions:
            if direction == 'forward':
                action_list_items.append("   - navigate to forward - continue straight ahead")
            elif direction == 'left':
                action_list_items.append("   - navigate to left - turn left and go forward")
            elif direction == 'right':
                action_list_items.append("   - navigate to right - turn right and go forward")
            elif direction == 'behind':
                action_list_items.append("   - navigate to behind - turn around and go forward")
            
            action_desc_parts.append("navigate to " + direction)
            
        action_list_str = "\n".join(action_list_items)
        action_desc_str = "|".join(action_desc_parts)

        # Build current views for prompt
        num_frames = len(panorama_frames)
        for view in view_definitions:
            angle = view['angle']
            if num_frames > 0:
                frame_idx = int((angle / 360.0) * num_frames) % num_frames
            else:
                frame_idx = 0

            if frame_idx < len(panorama_frames):
                rgb_image = panorama_frames[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                current_views.append({"type": "text", "text": view['label']})
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})

        # 4. Construct Prompt
        backtrack_point = visited_targets_up_to_backtrack[-1]
        previous_action = backtrack_point.get('direction_decision', 'unknown')

        # Generate negative constraints based on target object
        target_objects = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
        target = None
        instruction_lower = instruction.lower()
        for obj in target_objects:
            obj_name = obj.replace('_', ' ')
            if obj_name in instruction_lower or obj in instruction_lower:
                target = obj
                break
        
        negative_constraints = ""
        if self.use_negative_constraints:
            if target:
                other_objects = [obj.replace('_', ' ') for obj in target_objects if obj != target]
                negative_constraints = f"- When looking for {target.replace('_', ' ')}, do not confuse it with other objects: {', '.join(other_objects)}."

        prompt = self.P.LA_PROMPT_BACKTRACK_REPLAN.format(
            instruction=instruction,
            action_list=action_list_str,
            action_desc=action_desc_str,
            previous_action=previous_action,
            negative_constraints=negative_constraints
        )

        content = [{"type": "text", "text": "Navigation History:"}]
        content.extend(history_content)
        content.append({"type": "text", "text": "\nPrevious Trajectory (Path taken from here):"})
        content.extend(failed_path_content)
        content.append({"type": "text", "text": "\nCurrent 4-directional views at Backtrack Waypoint:"})
        content.extend(current_views)
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        self.logger.info("Replanning at backtrack point...")
        
        log_path = None
        if self.debug_logging and episode_id is not None and step is not None:
             import time
             log_path = os.path.join(self.log_dir, str(episode_id), f"step_{step}", f"la_replan_{time.time()}")

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=2048,
            temperature=0.7,
            use_la=True,
            log_path=log_path,
        )
        
        self.logger.info(f"Replan Output: {output_text}")
        
        # Parse JSON
        import json
        import re
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            try:
                response = json.loads(json_match.group())
                self.stair = response.get('stair', False) # Update stair status
                
                # Handle stop signal
                stop_signal = response.get('stop', False)
                action = response.get('action', 'navigate to forward').lower()
                
                if action == 'stop' or 'stop' in action.split():
                    stop_signal = True
                    if action == 'stop':
                        action = 'navigate to forward'
                
                response['stop_signal'] = stop_signal
                response['action'] = action
                
                if 'left' in action: return 'left', response
                if 'right' in action: return 'right', response
                if 'behind' in action: return 'behind', response
                return 'forward', response
            except:
                pass
        
        return 'forward', {}

    def reset(self):
        # self.model.reset_stats()
        self.stair = False
        self.todo_list = None
        self.last_object = ''

    def query_llm_oracle(self, panorama_frames, instruction):
        """EQA Oracle: answer the EQA question from 4 panorama views at the destination.

        Only invoked when task_type == 'EQA'.
        """
        content = [{
            'type': 'text',
            'text': self.P.ORACLE_SYSTEM_PROMPT.format(instruction=instruction),
        }]

        try:
            content.append({"type": "text", "text": "--- DESTINATION PANORAMA (Current Surroundings) ---"})
            view_definitions = [
                {'angle': 0, 'label': 'View 1: Forward'}, {'angle': 90, 'label': 'View 2: Left'},
                {'angle': 180, 'label': 'View 3: Behind'}, {'angle': 270, 'label': 'View 4: Right'}
            ]
            num_frames = len(panorama_frames)
            for view in view_definitions:
                frame_idx = int((view['angle'] / 360.0) * num_frames) % num_frames if num_frames > 0 else 0
                if frame_idx < len(panorama_frames):
                    raw = panorama_frames[frame_idx]['rgb']
                    img = Image.fromarray(raw) if isinstance(raw, np.ndarray) else raw
                    content.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{self.img_to_base64(img)}"}})
                    content.append({"type": "text", "text": view['label']})

            prompt = self.P.EQA_QA_PROMPT.format(instruction=instruction)
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=1024,
                temperature=0.7,
                use_la=True,
            )
            self.logger.info(f"Oracle QA Output: {output_text}")

            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group()).get('answer', 'null')

        except Exception as e:
            self.logger.error(f"Error in oracle QA: {e}")

        return "null"
