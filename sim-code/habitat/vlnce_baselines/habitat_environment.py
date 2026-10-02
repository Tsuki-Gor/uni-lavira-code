"""Concrete Habitat environment, mapping and local control without LaViRA."""

import copy
import gzip
import json
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict

import numpy as np
from PIL import Image
from fastdtw import fastdtw
from skimage.morphology import binary_closing
from torch import Tensor
from torchvision import transforms

from habitat import logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.environments import get_env_class

# Import Habitat visualization utilities

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.env.env_utils import construct_envs
from vlnce_baselines.utils.misc import get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.utils.constant import base_classes, map_channels
from .utils.visualization import LaViRAVisualizer

import sys
# Dynamic path resolution for Docker/local compatibility
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

NAVDP_ROOT = os.path.join(PROJECT_ROOT, "navdp")
if NAVDP_ROOT not in sys.path:
    sys.path.append(NAVDP_ROOT)

# iplanner was removed from the codebase. The guards below leave downstream
# `if self.use_iplanner:` branches unreachable (default USE_IPLANNER=False).
IPlannerAgent = None

try:
    from policy_agent import NavDP_Agent
except ImportError as _e:
    logger.warning(f"NavDP import failed ({_e}); --use-navdp will fail loudly when used.")
    NavDP_Agent = None

import warnings

warnings.filterwarnings('ignore')


class HabitatEnvironment:
    def __init__(self, config, r2r, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        self.r2r = r2r
        self.device = get_device(config.TORCH_GPU_ID)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.config = config
        self.map_args = config.MAP
        self.resolution = config.MAP.MAP_RESOLUTION
        self.width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.max_step = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution)

        self.trans = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(
                                             (self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH),
                                             interpolation=Image.NEAREST)
                                         ])

        self.classes = []
        self.current_episode_id = None
        self.floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)


        self.visualize = True
        _rgb_base = getattr(config, 'RGB_SAVE_DIR', './saved_rgb_images')
        # Namespace by exp_name (the per-run timestamp appended to EVAL_CKPT_PATH_DIR
        # in run_mp) so concurrent tasks / re-runs never overwrite each other's frames.
        _exp_name = os.path.basename(config.EVAL_CKPT_PATH_DIR.rstrip('/'))
        self.save_dir = os.path.join(_rgb_base, _exp_name) if _exp_name else _rgb_base
        # When DEBUG_LOGGING is on, also forward the debug_log_dir to the visualizer
        # so per-step topdown_map.png + rgb_obs.png land alongside the LA/VA prompts.
        _debug_logging = getattr(config, 'DEBUG_LOGGING', False)
        _debug_log_dir = getattr(config, 'DEBUG_LOG_DIR', 'logs/debug_logs') if _debug_logging else None
        self.visualizer = LaViRAVisualizer(None, self.visualize, self.save_dir, self.width, self.height,
                                           debug_log_dir=_debug_log_dir)

        self.current_step = 0  # Map update and recording step.
        
        self.config.defrost()
        self.config.MAP.DEVICE = self.config.TORCH_GPU_ID
        self.config.MAP.HFOV = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.config.MAP.AGENT_HEIGHT = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        self.config.MAP.NUM_ENVIRONMENTS = self.config.NUM_ENVIRONMENTS
        self.config.MAP.RESULTS_DIR = self.config.RESULTS_DIR
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.config.freeze()

        self.task_type = getattr(config, 'TASK_TYPE', 'VLN')
        self.terrain_hint = False

    def initialize_controllers(self):
        """Initialize the selected local planner; no navigation model is created."""
        config = self.config
        # The IPlanner backend was removed; this flag stays False so the inert
        # `if self.use_iplanner:` guards below never fire. Use FMM or NavDP instead.
        self.use_iplanner = False
        if self.use_iplanner:
            intrinsics = self._get_camera_intrinsics()
            intrinsics_tensor = torch.from_numpy(intrinsics).float().to(self.device)
            
            # Construct dynamic paths for iplanner model and config
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "iplanner", "plannernet.pt")
            config_path = os.path.join(project_root, "iplanner", "configs", "iplanner.yaml")
            
            logger.info(f"Loading iplanner from: {model_path} with config: {config_path}")
            
            self.iplanner_agent = IPlannerAgent(
                image_intrinsic=intrinsics_tensor,
                model_path=model_path,
                model_config_path=config_path,
                device=self.device
            )
            self.navdp_traj_global = None
            self.iplanner_update_freq = 5
            self.last_iplanner_step = -999


        # Initialize NavDP Agent
        self.use_navdp = getattr(config, 'USE_NAVDP', False)
        # logger.info(f"DEBUG: USE_NAVDP is set to {self.use_navdp}")
        if self.use_navdp:
            if NavDP_Agent is None:
                raise ImportError(
                    "USE_NAVDP=True but NavDP_Agent failed to import. "
                    "Likely missing `diffusers` (see navdp/requirements.txt). "
                    "Run: pip install diffusers==0.33.1"
                )
            intrinsics = self._get_camera_intrinsics()
            model_path = os.path.join(NAVDP_ROOT, "navdp-cross-modal.ckpt")
            # logger.info(f"Loading NavDP from: {model_path}")

            self.navdp_agent = NavDP_Agent(
                image_intrinsic=intrinsics,
                navi_model=model_path,
                device=self.device
            )
            self.navdp_agent.reset(batch_size=1, threshold=0.1)

        self.use_fmm = getattr(config, 'USE_FMM', True)
        
        # Validation Logic
        if self.use_navdp and self.use_iplanner:
             raise ValueError("Configuration Error: use_navdp and use_iplanner cannot both be True.")
        
        if not self.use_fmm and not self.use_navdp and not self.use_iplanner:
            raise ValueError("Configuration Error: use_fmm, use_navdp, and use_iplanner cannot all be False.")

    def _init_envs(self) -> None:
        # logger.info("start to initialize environments")

        # Check if we have specific allowed episodes loaded from assignment file
        specific_episodes = getattr(self, 'allowed_episodes', None)
        if specific_episodes:
            # self.allowed_episodes is a set of tuples (id, scene_id)
            specific_episodes = list(specific_episodes)

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
            specific_episodes_allowed=specific_episodes,
        )
        logger.info(f"local rank: {self.local_rank}, num of episodes: {self.envs.number_of_episodes}")
        self.detected_classes = OrderedSet()

    def _collect_val_traj(self) -> None:
        if not self.r2r:
            role = self.config.TASK_CONFIG.DATASET.ROLES
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        
        gt_path = None
        if self.r2r:
            gt_path = self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)
        else:
            gt_path = self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split, role=role[0])
            
        if os.path.exists(gt_path):
            with gzip.open(gt_path) as f:
                self.gt_data = json.load(f)
        else:
            self.gt_data = None
            logger.warning(f"GT data not found at {gt_path}, skipping NDTW calculation.")

    def _calculate_metric(self, infos: List, submitted_answer: str = None, is_timeout: bool = False):
        """Dispatch metric calculation by task_type."""
        if self.task_type == "EQA":
            return self._save_eqa_metric(infos, submitted_answer=submitted_answer, is_timeout=is_timeout)
        return self._calculate_trajectory_metric(infos)

    def _save_eqa_metric(self, infos: List[Dict], submitted_answer: str = None, is_timeout: bool = False):
        if not infos:
            return
        info = infos[0]
        curr_eps = self.envs.current_episodes()
        ep_id = curr_eps[0].episode_id
        if ep_id in getattr(self, 'state_eps', {}):
            return

        question_text = curr_eps[0].question.question_text
        ground_truth_answer = curr_eps[0].question.answer_text
        predicted_answer = submitted_answer
        # The complete task already supplied any history-based fallback answer.

        try:
            trajectory_list = info.get('position', {}).get('position', [])
            trajectory_json_string = json.dumps(trajectory_list)
        except Exception as e:
            logger.error(f"Failed to serialize trajectory: {e}")
            trajectory_json_string = "[]"

        metric = {
            'success': 0.0,
            'steps_taken': info.get('steps_taken', 0),
            'episode_id': ep_id,
            'question': question_text,
            'ground_truth': ground_truth_answer,
            'predicted_answer': predicted_answer if predicted_answer is not None else "TIMEOUT (No Answer)",
            'trajectory_json': trajectory_json_string,
        }

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.EVAL_CKPT_PATH_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )

        # Timeout only affects the recorded step count; the answer (forced
        # above if the agent never stopped) is still scored by exact match.
        # Only a genuinely missing answer scores 0.
        if is_timeout:
            metric['steps_taken'] = getattr(self, 'max_step', info.get('steps_taken', 0))

        if predicted_answer is not None:
            try:
                clean_gt = str(ground_truth_answer).strip()
                clean_pred = str(predicted_answer).strip()
                metric['success'] = 1.0 if clean_pred == clean_gt else 0.0
                logger.info(f"--- EQA JUDGMENT (Episode {ep_id}{' / TIMEOUT' if is_timeout else ''}) ---")
                logger.info(f"  Ground Truth    : {ground_truth_answer!r}")
                logger.info(f"  Submitted Answer: {predicted_answer!r}")
                logger.info(f"  SUCCESS : {metric['success']}")
            except Exception as e:
                logger.error(f"EQA judgement failure: {e}")
                metric['success'] = 0.0
        else:
            logger.info(f"--- Episode {ep_id} TIMEOUT, no answer available. Success: 0.0 ---")

        self.state_eps[ep_id] = metric
        try:
            with open(fname, "w") as f:
                json.dump(self.state_eps, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics to {fname}: {e}")

    def _calculate_trajectory_metric(self, infos: List):
        curr_eps = self.envs.current_episodes()
        info = infos[0]
        ep_id = curr_eps[0].episode_id

        metric = {}
        metric['steps_taken'] = info.get('steps_taken', 0)
        
        # Get success distance from config
        success_distance = getattr(self.config.TASK_CONFIG.TASK, 'SUCCESS_DISTANCE', 
                                 getattr(self.config.TASK_CONFIG.TASK.SUCCESS, 'SUCCESS_DISTANCE', 3.0))

        if self.gt_data is not None and str(ep_id) in self.gt_data:
            gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
            pred_path = np.array(info['position']['position'])
            distances = np.array(info['position']['distance'])
            gt_length = distances[0]
            dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
            
            metric['distance_to_goal'] = distances[-1]
            metric['success'] = 1. if distances[-1] <= success_distance else 0.
            metric['oracle_success'] = 1. if (distances <= success_distance).any() else 0.
            metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
            metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
            metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
            metric['sdtw'] = metric['ndtw'] * metric['success']
        else:
            # Use environment provided metrics if available
            metric['distance_to_goal'] = info.get('distance_to_goal', 0.0)
            metric['success'] = info.get('success', 0.0)
            metric['oracle_success'] = info.get('oracle_success', 0.0)
            metric['path_length'] = info.get('path_length', 0.0)
            metric['spl'] = info.get('spl', 0.0)
            metric['ndtw'] = info.get('ndtw', 0.0)
            metric['sdtw'] = info.get('sdtw', 0.0)
            
        self.state_eps[ep_id] = metric
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        logger.info(f'ep{ep_id}:{self.state_eps[ep_id]}')

        # Visualize trajectory
        self.draw_agent_pos_and_ref_path(success=bool(metric['success']))

        # Rename the episode directory with success status
        try:
            is_success = bool(metric['success'])
            status_suffix = "_success" if is_success else "_failed"
            
            # The current save directory for this episode
            current_dir = os.path.join(self.save_dir, str(ep_id))
            
            if os.path.exists(current_dir):
                # The new directory name
                new_dir = os.path.join(self.save_dir, f"{ep_id}{status_suffix}")
                
                if os.path.exists(new_dir):
                    import shutil
                    shutil.rmtree(new_dir)
                
                os.rename(current_dir, new_dir)
                logger.info(f"Renamed episode directory to: {new_dir}")
        except Exception as e:
            logger.info(f"Error renaming episode directory: {e}")

    def _initialize_policy(self) -> None:
        # logger.info("start to initialize policy")
        self.segment_module = GroundedSAM(self.config, self.device)
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        self.visualizer.update_map(self.mapping_module)

        self.policy = FusionMapPolicy(self.config, self.mapping_module.map_shape[0])
        self.policy.reset()

    def _concat_obs(self, obs: Observations) -> np.ndarray:
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)  # (h, w, c)->(c, h, w)

        return state

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8)  # [3, h, w]
        rgb = rgb[:, :, ::-1]  # RGB to BGR
        depth = state[:, :, 3:4]  # [1, h, w]
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH

        sem_seg_pred = self._get_sem_pred(rgb)  # [num_detected_classes, h, w]
        depth = self._preprocess_depth(depth, min_depth, max_depth)  # [1, h, w]

        """
        ds: Downscaling factor
        args.env_frame_width = 640, args.frame_width = 160
        """
        ds = env_frame_width // self.map_args.FRAME_WIDTH  # ds = 4
        if ds != 1:
            rgb = np.asarray(self.trans(rgb.astype(np.uint8)))  # resize
            depth = depth[ds // 2::ds, ds // 2::ds]  # down scaling start from 2, step=4
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)  # recover depth.shape to (height, width, 1)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)  # (4+num_detected_classes, h, w)

        return state

    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray:
        """
        mask.shape=[num_detected_classes, h, w]
        labels looks like: ["kitchen counter 0.69", "floor 0.37"]
        """
        cls2 = self.classes.copy()
        if getattr(self, 'terrain_hint', False):
            cls2 = ["stairs"]
        elif self.current_step <= 12:
            cls2.append('stairs')
        masks, labels, annotated_images, self.current_detections = \
            self.segment_module.segment(rgb, classes=cls2)
        if self.visualize:
            pass
            # save_path = os.path.join(self.save_dir, str(self.current_episode_id), f'step{self.current_step}_mask.png')
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # cv2.imwrite(save_path, annotated_images)
        self.mapping_module.rgb_vis = annotated_images
        assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"
        # logger.info("current step detected classes (before filtering): ", labels)

        class_names = self._process_labels(labels)
        masks = self._process_masks(masks, class_names)

        return masks.transpose(1, 2, 0)

    def _process_labels(self, labels: List[str]) -> List:
        class_names = []
        for label in labels:
            class_name = " ".join(label.split(' ')[:-1])
            class_names.append(class_name)
            self.detected_classes.add(class_name)

        return class_names

    def _process_masks(self, masks: np.ndarray, labels: List[str]):
        """Since we are now handling the open-vocabulary semantic mapping problem,
        we need to maintain a mask tensor with dynamic channels. The idea is to combine
        all same class tensors into one tensor, then let the "detected_classes" to 
        record all classes without duplication. Finally we can use each class's index
        in the detected_classes to determine as it's channel in the mask tensor.
        The organization of mask is similar to chaplot's Sem_Exp, please refer to this link:
        https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/agents/utils/semantic_prediction.py#L41
        
        Args:
            masks (np.ndarray): shape:(c,h,w), each instance(even the same class) has one channel
            labels (List[str]): masks' corresponding labels. len(masks) = len(labels)

        Returns:
            final_masks (np.ndarray): each mask will find their channel in self.detected_classes.
            len(final_masks) = len(self.detected_classes)
        """
        if masks.shape[0] > 0:  # Check if there are any masks
            same_label_indexs = defaultdict(list)
            for idx, item in enumerate(labels):
                same_label_indexs[item].append(idx)  # dict {class name: [idx]}
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)

            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]

            """
            max_idx = max(idx) + 1, attention: remember to add one becaure index start from 0
            init final masks as [max_idx + 1, h, w]; add not_a_category channel at last
            """
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), self.height, self.width))

        return final_masks

    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        # Preprocesses a depth map by handling missing values, removing outliers, and scaling the depth values.
        # logger.info('max:',depth.max())
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99  # turn too far pixels to invalid
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 1.0  # then turn all invalid pixels to vision_range(100)
        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0

        return depth

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        concated_obs = self._concat_obs(obs)
        state = self._preprocess_state(concated_obs)

        return state  # state.shape=(c,h,w)

    def _batch_obs(self, n_obs: List[Observations]) -> Tensor:
        n_states = [self._preprocess_obs(obs) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state,
                                 [(0, max_channels - state.shape[0]),
                                  (0, 0),
                                  (0, 0)],
                                 mode='constant')
                          for state in n_states], axis=0)

        return torch.from_numpy(batch).to(self.device)

    def _process_one_step_floor(self, one_step_full_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        # logger.info(f'{navigable_index}, {not_navigable_index}')
        one_step_full_map = remove_small_objects(one_step_full_map.astype(bool), min_size=64)

        obstacles = one_step_full_map[0, ...].astype(bool)
        explored_area = one_step_full_map[1, ...].astype(bool)

        objects = np.sum(one_step_full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)
        navigable = np.logical_or.reduce(one_step_full_map[map_channels:, ...][navigable_index])
        # stairs should remain navigable even if overlapped with objects
        # navigable = np.logical_or(navigable, stairs_mask)
        navigable = np.logical_and(navigable, np.logical_not(objects))

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        # free_mask = np.logical_or(free_mask, stairs_mask)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=disk(kernel_size))

        return floor

    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int = 3) -> tuple:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        full_map = remove_small_objects(full_map.astype(bool), min_size=64)

        obstacles = full_map[0, ...].astype(bool)
        explored_area = full_map[1, ...].astype(bool)

        objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)

        selem = disk(3)
        obstacles_closed = binary_closing(obstacles, footprint=selem)
        objects_closed = binary_closing(objects, footprint=selem)
        navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index])
        # stairs should remain navigable even if overlapped with objects
        # navigable = np.logical_or(navigable, stairs_mask)
        navigable = np.logical_and(navigable, np.logical_not(objects))
        navigable_closed = binary_closing(navigable, footprint=selem)

        untraversable = np.logical_or(objects_closed, obstacles_closed)
        # ensure stairs override untraversable
        untraversable[navigable_closed == 1] = 0
        # untraversable[stairs_mask == 1] = 0
        untraversable = remove_small_objects(untraversable, min_size=64)
        untraversable = binary_closing(untraversable, footprint=disk(3))
        traversable = np.logical_not(untraversable)

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        # free_mask = np.logical_or(free_mask, stairs_mask)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=selem)
        traversable = np.logical_or(floor, traversable)

        explored_area = binary_closing(explored_area, footprint=selem)
        contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
        frontiers = np.logical_and(floor, image)
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

        return traversable, floor, frontiers.astype(np.uint8)

    def _maps_initialization(self):
        obs = self.envs.reset()  # type(obs): list
        
        # Handle instruction extraction for different tasks (VLN / ObjectNav / EQA)
        current_episode = self.envs.current_episodes()[0]
        if self.task_type == "EQA" and hasattr(current_episode, 'question'):
            self.instruction = current_episode.question.question_text
        elif 'instruction' in obs[0]:
            self.instruction = obs[0]['instruction']['text']
        elif hasattr(current_episode, 'object_category'):
            self.instruction = f"Find the {current_episode.object_category}"
        elif 'objectgoal' in obs[0]:
            self.instruction = f"Find the object with ID {obs[0]['objectgoal'][0]}"
        else:
            self.instruction = "Explore the environment"
            
        self.destination = "goal"
        self.classes = self.base_classes.copy()
        self.current_episode_id = self.envs.current_episodes()[0].episode_id

        self.visualizer._save_rgb_frame(obs[0], 0, None, self.current_episode_id)

        # Initialize trajectory tracking
        self.pos_list = []
        # Get reference path if available
        self.ref_path = []
        if self.gt_data is not None and str(self.current_episode_id) in self.gt_data:
            self.ref_path = self.gt_data[str(self.current_episode_id)]['locations']

        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        batch_obs = self._batch_obs(obs)
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
        self.mapping_module(batch_obs, poses, self.current_step)
        full_map, full_pose, _ = self.mapping_module.update_map(0, self.detected_classes, self.current_episode_id)
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)

        return obs, full_pose

    def reset(self) -> None:
        self.classes = []
        self.detected_classes = OrderedSet()
        self.floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)

        self.current_step = 0

        self.policy.reset()
        self.mapping_module.reset()
        self.terrain_hint = False

    def _get_camera_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix for depth projection"""
        hfov = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        width = self.width
        height = self.height
        vfov = 2 * np.arctan(height / width * np.tan(hfov / 2))

        fx = width / (2 * np.tan(np.deg2rad(hfov / 2)))
        fy = height / (2 * np.tan(np.deg2rad(vfov / 2)))
        cx = width / 2
        cy = height / 2

        intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])
        return intrinsics

    def draw_agent_pos_and_ref_path(self, success: bool = False):
        """
        Visualize agent path, reference path, and target object/viewpoints using Matplotlib.
        """
        if not getattr(self, "visualize", False):
            return None

        pos_list = getattr(self, "pos_list", None) or []
        ref_path = getattr(self, "ref_path", None) or []
        
        # Get target object and viewpoints
        current_episode = self.envs.current_episodes()[0]
        goals = getattr(current_episode, 'goals', [])
        
        # Extract data for plotting
        # Habitat coordinates: Y is up. We visualize X-Z plane (Top-down).
        agent_x = [p[0] for p in pos_list]
        agent_z = [p[2] for p in pos_list]
        
        ref_x = [p[0] for p in ref_path]
        ref_z = [p[2] for p in ref_path]
        
        object_centers = []
        closest_viewpoints = []
        all_viewpoints = []
        radius_list = []
        
        for goal in goals:
            if hasattr(goal, 'position'):
                obj_pos = goal.position
                object_centers.append(obj_pos)
                # Check for radius
                if hasattr(goal, 'radius') and goal.radius is not None:
                     radius_list.append((obj_pos, goal.radius))
            
                # Find closest viewpoint for this goal
                if hasattr(goal, 'view_points') and len(goal.view_points) > 0:
                    best_vp = None
                    min_dist = float('inf')
                    
                    for vp in goal.view_points:
                        if hasattr(vp, 'agent_state') and hasattr(vp.agent_state, 'position'):
                            vp_pos = vp.agent_state.position
                            all_viewpoints.append(vp_pos)
                            # Calculate Euclidean distance in 3D
                            dist = np.linalg.norm(np.array(vp_pos) - np.array(obj_pos))
                            if dist < min_dist:
                                min_dist = dist
                                best_vp = vp_pos
                    
                    if best_vp is not None:
                        closest_viewpoints.append(best_vp)

        # Plotting
        plt.figure(figsize=(10, 8))
        
        # Plot Object Centers
        for obj in object_centers:
            plt.scatter(obj[0], obj[2], c='red', s=200, marker='*', label='Object Center')
            
        # Plot Radius zones
        for center, rad in radius_list:
             circle = plt.Circle((center[0], center[2]), rad, color='green', fill=True, alpha=0.2)
             plt.gca().add_patch(circle)

        # Plot ALL ViewPoints (but circle only closest)
        if all_viewpoints:
            vp_x_all = [vp[0] for vp in all_viewpoints]
            vp_z_all = [vp[2] for vp in all_viewpoints]
            plt.scatter(vp_x_all, vp_z_all, c='green', s=30, alpha=0.6, label='ViewPoints')

        # Draw Success Zones (1.0m) for closest viewpoints only
        if closest_viewpoints:
            vp_x_closest = [vp[0] for vp in closest_viewpoints]
            vp_z_closest = [vp[2] for vp in closest_viewpoints]
            
            for vx, vz in zip(vp_x_closest, vp_z_closest):
                circle = plt.Circle((vx, vz), 1.0, color='green', fill=False, linestyle='--', alpha=0.5)
                plt.gca().add_patch(circle)
        
        # Plot Reference Path
        if ref_x:
            plt.plot(ref_x, ref_z, c='green', linewidth=2, label='Reference Path', linestyle='-')
            # Start and End of Reference Path
            plt.scatter(ref_x[0], ref_z[0], c='orange', s=100, marker='o', label='Ref Start')
            plt.scatter(ref_x[-1], ref_z[-1], c='blue', s=100, marker='X', label='Ref Goal')

        # Plot Agent Trajectory
        if agent_x:
            plt.plot(agent_x, agent_z, c='red', linewidth=2, label='Agent Path', linestyle='-')
            if agent_x:
                plt.scatter(agent_x[0], agent_z[0], c='blue', s=100, marker='^', label='Agent Start')
                plt.scatter(agent_x[-1], agent_z[-1], c='red', s=80, marker='v', label='Agent End')

        plt.title(f'Episode {getattr(self, "current_episode_id", "unknown")} Trajectory')
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters) - Top Down View')
        
        # Avoid duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        
        episode_id = str(getattr(self, "current_episode_id", "unknown"))
        # out_dir = os.path.join(self.save_dir, episode_id)
        # Modified to save to traj_viz directory
        out_dir = os.path.join(self.save_dir, "traj_viz")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"traj_viz_success_{success}_{episode_id}.png")
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved trajectory visualization to: {out_path}")

        return out_path
