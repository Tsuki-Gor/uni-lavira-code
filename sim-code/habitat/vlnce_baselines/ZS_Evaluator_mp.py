"""Legacy evaluation entrypoint assembling Habitat and the shared LaViRA core."""

import json
import os
import time

from tqdm import tqdm
from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry

from .habitat_environment import HabitatEnvironment
from .utils.map_utils import torch, calculate_displacement, collision_check_fmm
from .utils.depth_utils import get_world_xz_from_pixel


@baseline_registry.register_trainer(name="ZS-Evaluator-mp")
class ZeroShotVlnEvaluatorMP(HabitatEnvironment, BaseTrainer):
    def __init__(self, config, r2r, segment_module=None, mapping_module=None) -> None:
        super().__init__(config, r2r, segment_module, mapping_module)
        from .agent import VLMReasoningAgent

        self.visited_targets = []
        self.history_images = []
        self.target_reached_threshold = getattr(config, 'TARGET_REACHED_THRESHOLD', 75.0 / self.resolution)
        self.total_backtracks = 0
        self.total_waypoints = 0
        self.use_continuous_history = getattr(config.EVAL, 'USE_CONTINUOUS_HISTORY', True)
        self.history_interval = getattr(config.EVAL, 'HISTORY_INTERVAL', 2)
        debug_logging = getattr(config, 'DEBUG_LOGGING', False)
        debug_log_dir = getattr(config, 'DEBUG_LOG_DIR', 'logs/debug_logs')
        self.use_todo_list = getattr(config.EVAL, 'USE_TODO_LIST', True)
        self.backtrack_second_chance = getattr(config.EVAL, 'BACKTRACK_SECOND_CHANCE', True)
        self.agent = VLMReasoningAgent(
            self.visualizer,
            task_type=self.task_type,
            use_guideline=True,
            use_working_memory=False,
            allow_move_behind=False,
            debug_logging=debug_logging,
            log_dir=debug_log_dir,
            use_todo_list=self.use_todo_list,
            backtrack_second_chance=self.backtrack_second_chance,
        )
        self.initialize_controllers()

    def reset(self) -> None:
        self.visited_targets = []
        self.history_images = []
        self.backtrack_steps = 0
        self.last_submitted_answer = None
        self._eqa_done_requested = False
        super().reset()
        self.agent.reset()

    def rollout(self, cancellation=None, on_progress=None):
        """
        Execute a whole episode using bounding box target navigation
        """
        from lavira.agent import LaViRAAgent
        from lavira.runtime import Task
        from navigation_runtime.mapped_simulator import MappedSimulatorRuntime

        if self.use_navdp:
            self.navdp_agent.reset_env(0)

        obs, full_pose = self._maps_initialization()

        # Check allowed episodes
        if getattr(self, 'allowed_episodes', None) is not None:
             current_ep = self.envs.current_episodes()[0]
             
             # Robust matching for (id, scene_id) to handle path differences
             is_allowed = False
             for allowed_id, allowed_scene_id in self.allowed_episodes:
                 if str(current_ep.episode_id) == allowed_id:
                     # Check Scene match (basename or suffix)
                     # allowed_scene_id is usually relative, current_ep.scene_id is absolute
                     if current_ep.scene_id.endswith(allowed_scene_id) or \
                        os.path.basename(current_ep.scene_id) == os.path.basename(allowed_scene_id):
                         is_allowed = True
                         break
             
             if not is_allowed:
                 ep_info = f"{current_ep.episode_id} ({os.path.basename(current_ep.scene_id)})"
                 logger.info(f"Skipping episode {ep_info} (Not in assignment)")
                 return

        runtime = MappedSimulatorRuntime(
            self, obs, full_pose, tensor_api=torch, actions=HabitatSimActions,
            project_pixel=get_world_xz_from_pixel, displacement=calculate_displacement,
            collision_check=collision_check_fmm, log=logger)
        core = LaViRAAgent(
            self.agent, "habitat", use_continuous_history=self.use_continuous_history,
            history_interval=self.history_interval, target_reached_cells=self.target_reached_threshold)
        task = Task(
            self.instruction, {"VLN": "vln", "OBJNAV": "object_nav", "EQA": "eqa"}[self.task_type],
            max_steps=max(0, self.max_step), id=self.current_episode_id,
            answer_from_history=self.current_episode_id not in getattr(self, 'state_eps', {}))
        try:
            self.result = core.run(task, runtime, cancellation=cancellation, on_progress=on_progress)
        finally:
            self.visited_targets = core.state.visited_targets
            self.history_images = core.state.history_images
            self.total_waypoints += core.state.total_waypoints
            self.total_backtracks += core.state.total_backtracks
        self.last_submitted_answer = self.result.answer
        self._calculate_metric(
            runtime.infos, submitted_answer=self.result.answer,
            is_timeout=self.result.reason == "budget_exhausted" or runtime.panorama_interrupted)


    def eval(self):
        # Load detailed assignments if available. File is namespaced by exp_name
        # (derived from the unique EVAL_CKPT_PATH_DIR) to allow parallel runs.
        exp_name = os.path.basename(self.config.EVAL_CKPT_PATH_DIR.rstrip('/'))
        assignment_file = f"data/logs/running_log/worker_{self.local_rank}_assignments_{exp_name}.json"
        if not os.path.exists(assignment_file):
            # Backward compat: fall back to the un-namespaced filename
            assignment_file = f"data/logs/running_log/worker_{self.local_rank}_assignments.json"
        if os.path.exists(assignment_file):
            with open(assignment_file, 'r') as f:
                assignments = json.load(f)
                # Set of (id, scene_id) tuples for fast lookup
                self.allowed_episodes = set((str(item['id']), item['scene_id']) for item in assignments)
            logger.info(f"Loaded {len(self.allowed_episodes)} specific episodes to process from {assignment_file}")

        # Sampling logic
        if self.config.EVAL.EPISODE_COUNT > -1:
            pass # Sampling is handled in run_mp.py, here we just use the assigned episodes


        # self._set_eval_config()
        self._init_envs()
        self._collect_val_traj()
        self._initialize_policy()
        self.agent.reset()
        self.terrain_hint = False

        # Reset model usage stats
        self.agent.model.reset_stats()

        eps_to_eval = sum(self.envs.number_of_episodes)

        self.state_eps = {}
        t1 = time.time()
        
        logger.info(f"Worker {self.local_rank}: Starting eval loop with {eps_to_eval} episodes")
        
        for i in tqdm(range(eps_to_eval), desc=f"Worker {self.local_rank}"):
            self.rollout()
            self.reset()

        self.envs.close()

        # Print final statistics
        logger.info("=== FINAL MODEL USAGE STATISTICS ===")
        final_stats = self.agent.model.print_usage_stats()

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)

        # Calculate and print average metrics
        # avg_metrics = defaultdict(float)
        # num_episodes = len(self.state_eps)
        # if num_episodes > 0:
        #     for metrics in self.state_eps.values():
        #         for k, v in metrics.items():
        #             if isinstance(v, (int, float)):
        #                 avg_metrics[k] += v

        #     for k in avg_metrics:
        #         avg_metrics[k] /= num_episodes

        #     logger.info("Average Metrics:")
        #     logger.info(json.dumps(dict(avg_metrics), indent=2))
        # else:
        #     logger.info("No episodes evaluated.")

        # Save model usage stats to separate file
        stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                   f"model_usage_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(stats_fname, "w") as f:
            json.dump(final_stats, f, indent=2)
        logger.info(f"Model usage statistics saved to: {stats_fname}")

        # Save navigation statistics (backtracks, waypoints)
        nav_stats = {
            'total_backtracks': self.total_backtracks,
            'total_waypoints': self.total_waypoints
        }
        nav_stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                       f"nav_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(nav_stats_fname, "w") as f:
            json.dump(nav_stats, f, indent=2)
        logger.info(f"Navigation statistics saved to: {nav_stats_fname}")

        t2 = time.time()
        logger.info(f"time: {t2 - t1}s")
        logger.info("test time: %d", t2 - t1)


    def eval_dynamic(self, ep_queue):
        """Dynamic work-stealing eval: pop episodes from a shared mp.Queue.

        Each worker reuses its trainer/policy/model across episodes but
        rebuilds the habitat env per episode (so EPISODES_ALLOWED can be
        re-targeted to a single id). The shared queue lets fast workers
        grab the next pending episode instead of being stuck with a fixed
        slice.
        """
        import queue as _queue
        # Heavy initialisation done once per worker
        self._initialize_policy()
        self._collect_val_traj()
        self.agent.model.reset_stats()
        self.state_eps = {}
        t1 = time.time()
        ep_count = 0
        logger.info(f"Worker {self.local_rank}: dynamic queue eval starting")

        while True:
            try:
                ep_info = ep_queue.get(timeout=2)
            except _queue.Empty:
                logger.info(f"Worker {self.local_rank}: queue empty, exiting after {ep_count} eps")
                break
            ep_id = str(ep_info['id'])
            scene_id = ep_info['scene_id']

            # Re-target the dataset filter to a single ep id and rebuild env.
            self.config.defrost()
            self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [ep_id]
            self.config.freeze()
            self.allowed_episodes = {(ep_id, scene_id)}
            if hasattr(self, 'envs') and self.envs is not None:
                try:
                    self.envs.close()
                except Exception as e:
                    logger.warning(f"envs.close() failed: {e}")
            self._init_envs()
            self.agent.reset()
            self.terrain_hint = False

            try:
                self.rollout()
            except Exception as e:
                logger.error(f"Worker {self.local_rank} ep {ep_id} rollout failed: {e}")
            try:
                self.reset()
            except Exception as e:
                logger.error(f"Worker {self.local_rank} ep {ep_id} reset failed: {e}")
            ep_count += 1

        # Final cleanup + stats persistence (mirrors eval())
        if hasattr(self, 'envs') and self.envs is not None:
            try:
                self.envs.close()
            except Exception:
                pass

        final_stats = self.agent.model.print_usage_stats()
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        stem = f"{split}_r{self.local_rank}_w{self.world_size}"
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"stats_ep_ckpt_{stem}.json"), "w") as f:
            json.dump(self.state_eps, f, indent=2)
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"model_usage_stats_{stem}.json"), "w") as f:
            json.dump(final_stats, f, indent=2)
        with open(os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"nav_stats_{stem}.json"), "w") as f:
            json.dump({'total_backtracks': self.total_backtracks, 'total_waypoints': self.total_waypoints}, f, indent=2)
        logger.info(f"Worker {self.local_rank}: ran {ep_count} eps in {time.time() - t1:.0f}s")
