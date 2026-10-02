"""Exercise the concrete Habitat host with offline platform dependencies."""

import importlib
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NS = types.SimpleNamespace


def platform_modules(trace):
    """Replace unavailable simulator/networks, leaving project assembly code real."""
    modules = {}

    def module(name, **exports):
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            key = ".".join(parts[:index])
            if key == "vlnce_baselines":
                continue
            if key not in modules:
                modules[key] = types.ModuleType(key)
                modules[key].__path__ = []
        modules[name].__dict__.update(exports)
        return modules[name]

    class Trainer:
        def __init__(self):
            trace.append("trainer")

    class Visualizer:
        def __init__(self, *args, **kwargs):
            trace.append("visualizer")

        def update_map(self, mapping):
            trace.append("visualizer.map")

    class Mapping:
        map_shape = (20, 20)

        def __init__(self, *args):
            trace.append("mapping")

        def to(self, device):
            return self

        def eval(self):
            trace.append("mapping.eval")

        def reset(self):
            trace.append("mapping.reset")

    class Policy:
        def __init__(self, *args):
            trace.append("policy")

        def reset(self):
            trace.append("policy.reset")

    class NavDP:
        def __init__(self, **kwargs):
            trace.append("navdp")

        def reset(self, **kwargs):
            trace.append(("navdp.reset", kwargs))

    def construct_envs(config, env_class, **kwargs):
        trace.append(("envs", kwargs))
        return NS(number_of_episodes=[1], close=lambda: trace.append("envs.close"))

    torch = module("torch", Tensor=object, cuda=NS(is_available=lambda: False))
    module("matplotlib", use=lambda backend: None)
    module("matplotlib.pyplot")
    module("fastdtw", fastdtw=None)
    module("skimage.morphology", binary_closing=None)
    module("torchvision", transforms=NS(Compose=lambda items: items,
           ToPILImage=lambda: None, Resize=lambda *args, **kwargs: None))
    module("tqdm", tqdm=lambda items, **kwargs: items)
    module("habitat", logger=logging.getLogger("habitat-environment-test"))
    module("habitat_extensions.measures", NDTW=object)
    module("habitat.core.simulator", Observations=dict)
    module("habitat_baselines.common.base_trainer", BaseTrainer=Trainer)
    module("habitat_baselines.common.environments", get_env_class=lambda name: object)
    module("habitat.sims.habitat_simulator.actions", HabitatSimActions=NS())
    module("habitat_baselines.common.baseline_registry",
           baseline_registry=NS(register_trainer=lambda **kwargs: lambda cls: cls))
    module("vlnce_baselines.utils.map_utils", torch=torch,
           calculate_displacement=None, collision_check_fmm=None)
    module("vlnce_baselines.utils.data_utils", OrderedSet=set)
    module("vlnce_baselines.map.mapping", Semantic_Mapping=Mapping)
    module("vlnce_baselines.models.Policy", FusionMapPolicy=Policy)
    module("vlnce_baselines.env.env_utils", construct_envs=construct_envs)
    module("vlnce_baselines.env.environments")
    module("vlnce_baselines.utils.misc", get_device=lambda index: NS(type="cpu"))
    module("vlnce_baselines.map.semantic_prediction",
           GroundedSAM=lambda *args: trace.append("segmentation"))
    module("vlnce_baselines.utils.constant", base_classes=["chair"], map_channels={})
    module("vlnce_baselines.utils.depth_utils", get_world_xz_from_pixel=None)
    module("vlnce_baselines.utils.visualization", LaViRAVisualizer=Visualizer)
    module("policy_agent", NavDP_Agent=NavDP)
    return modules


def config(navdp, kind):
    return NS(
        TORCH_GPU_ID=0, NUM_ENVIRONMENTS=1, RESULTS_DIR="unused",
        EVAL_CKPT_PATH_DIR="unused/run", RGB_SAVE_DIR="unused", ENV_NAME="test",
        world_size=1, local_rank=0, USE_NAVDP=navdp, USE_FMM=not navdp, TASK_TYPE=kind,
        EVAL=NS(), MAP=NS(MAP_RESOLUTION=5, MAP_SIZE_CM=100, FRAME_HEIGHT=12, FRAME_WIDTH=16),
        TASK_CONFIG=NS(SIMULATOR=NS(RGB_SENSOR=NS(WIDTH=16, HEIGHT=12, HFOV=90),
                                   AGENT_0=NS(HEIGHT=1.5)),
                       ENVIRONMENT=NS(MAX_EPISODE_STEPS=80),
                       DATASET=NS(EPISODES_ALLOWED=["1"])),
        defrost=lambda: None, freeze=lambda: None,
    )


class HabitatEnvironmentTest(unittest.TestCase):
    def test_environment_initialization_has_no_lavira_dependency(self):
        import cv2
        from PIL import Image
        Image.init()
        for navdp in (False, True):
            for kind in ("VLN", "OBJNAV", "EQA"):
                with self.subTest(navdp=navdp, kind=kind), patch.dict(sys.modules):
                    for name in list(sys.modules):
                        if name.split(".")[0] in ("vlnce_baselines", "lavira", "openai"):
                            del sys.modules[name]
                    trace = []
                    sys.modules.update(platform_modules(trace))
                    sys.modules.update({name: None for name in
                                        ("lavira", "openai", "vlnce_baselines.agent")})
                    with patch.object(sys, "path", [str(ROOT / "sim-code/habitat")] + sys.path):
                        # Importing the real package still performs trainer registration.
                        host_class = importlib.import_module(
                            "vlnce_baselines.habitat_environment").HabitatEnvironment
                        host = host_class(config(navdp, kind), r2r=True)
                        host.initialize_controllers()
                        host._init_envs()
                        host._initialize_policy()
                        host.reset()
                        host.envs.close()
                        self.assertFalse(hasattr(host, "agent"))
                        self.assertFalse(hasattr(host, "visited_targets"))
                        self.assertFalse(hasattr(host, "history_images"))
                        self.assertEqual(host.task_type, kind)
                        self.assertFalse(host.terrain_hint)
                        expected = ["visualizer"]
                        if navdp:
                            expected += ["navdp", ("navdp.reset", {"batch_size": 1, "threshold": 0.1})]
                        expected += [("envs", {"auto_reset_done": False,
                                              "episodes_allowed": ["1"], "specific_episodes_allowed": None}),
                                     "segmentation", "mapping", "mapping.eval", "visualizer.map",
                                     "policy", "policy.reset", "policy.reset", "mapping.reset", "envs.close"]
                        self.assertEqual(trace, expected)

                        # Legacy trainer assembles the model between visualizer and controller,
                        # as before. The platform host itself never requests it.
                        trace.clear()
                        agent_module = types.ModuleType("vlnce_baselines.agent")
                        def make_agent(*args, **kwargs):
                            trace.append(("agent", kwargs))
                            return NS(reset=lambda: trace.append("agent.reset"))
                        agent_module.VLMReasoningAgent = make_agent
                        sys.modules[agent_module.__name__] = agent_module
                        entry = importlib.import_module("vlnce_baselines.ZS_Evaluator_mp")
                        trainer = entry.ZeroShotVlnEvaluatorMP(config(navdp, kind), r2r=True)
                        self.assertEqual(trainer.visited_targets, [])
                        self.assertEqual(trainer.history_images, [])
                        self.assertEqual(trace[:2], ["trainer", "visualizer"])
                        self.assertEqual(trace[2][0], "agent")
                        self.assertEqual(trace[2][1]["task_type"], kind)
                        if navdp:
                            self.assertEqual(trace[3], "navdp")
                        trainer._initialize_policy()
                        trainer.reset()
                        self.assertEqual(trace[-3:], ["policy.reset", "mapping.reset", "agent.reset"])


if __name__ == "__main__":
    unittest.main()
