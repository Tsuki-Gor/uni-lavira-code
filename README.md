# Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation

Uni-LaViRA is a **training-free** framework for **unified embodied navigation**: instead of fine-tuning on millions of robot data and thousands of GPU hours, it casts navigation as a three-layer *translation* — **language → vision → robot action** — where pretrained multimodal LLMs plan in language, ground in vision, and map the result to each robot's low-level controls. One architecture spans four navigation tasks and four heterogeneous robots, yet **matches or even surpasses** trained SOTA navigation foundation models.

![Uni-LaViRA teaser](assets/teaser.png)

## Release TODO List

- [x] Conda environment setup
- [x] Docker environment setup
- [x] Simulation evaluation code
    - [x] VLN — R2R-CE, RxR-CE
    - [x] ObjectNav — HM3D-v2, HM3D-OVON
    - [x] EQA — MP3D-EQA
    - [x] Aerial VLN — OpenUAV
- [x] Real-world deployment code
    - [x] Agilex Cobot Magic
    - [x] Unitree Go1
    - [x] Unitree G1
    - [x] Self-built UAV

## Repository Layout

| Path | Description |
|------|-------------|
| `lavira/` | Shared LaViRA clients, reasoning and task driver for Habitat, Go1, G1 and Cobot. |
| `navigation_runtime/` | Algorithm-neutral observations, goals and camera, gimbal and mapped-simulator execution. |
| `sim-code/habitat/` | Habitat-based simulation for VLN + ObjectNav + EQA. |
| `sim-code/airsim/`  | AirSim / TravelUAV-based simulation for aerial VLN. |
| `real-world-code/cobot_magic/` | Agilex Cobot Magic deployment. |
| `real-world-code/unitree_g1/`  | Unitree G1 humanoid deployment. |
| `real-world-code/unitree_go1/` | Unitree Go1 quadruped deployment. |
| `real-world-code/self_built_uav/` | Self-built UAV (ROS) deployment. |

## Task × Platform

| Task | Simulation | Real Robots |
|------|-----------|-------------|
| VLN        | `sim-code/habitat` | cobot_magic · unitree_g1 · unitree_go1 · self_built_uav |
| ObjectNav  | `sim-code/habitat` | cobot_magic · unitree_g1 · unitree_go1 · self_built_uav |
| EQA        | `sim-code/habitat` | cobot_magic · unitree_g1 · unitree_go1 · self_built_uav |
| Aerial VLN | `sim-code/airsim`  | self_built_uav |

Habitat and the three ground robots use the shared core for `vln`, `object_nav` and `eqa`, preserving their task prompts and platform policies. `interact` remains the platform's web/voice entrypoint.

## Quickstart

- **Run simulation evaluation** → `sim-code/habitat/README.md` (indoor) or `sim-code/airsim/README.md` (aerial).
- **Deploy on a real robot** → the README inside the matching `real-world-code/<robot>/` folder.

Environment setup, datasets, checkpoints, and run commands live in each platform's README. Platforms retain separate environments; Habitat, Go1, G1 and Cobot also install the shared package from the repository root (`pip install -e ../..` from their platform directory).

The shared-core [refactoring design](unified-lavira-agent.md) is maintained in Markdown; the HTML remains available for reference.

The shared core is being extracted in stages. See the [design](docs/lavira-core-design.md) and [migration status](docs/lavira-migration.md) for scope and validation. Offline core checks run from the repository root with `python -m unittest discover -s tests -v`; these checks do not connect to robots or model services.

The in-process task interface is documented in [using the shared core](docs/lavira-usage.md). Install the existing image dependencies to run image and navigation checks; otherwise those checks report skips.

## Citation

```bibtex
@article{ding2026unilavira,
  title={Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation},
  author={Ding, Hongyu and others},
  journal={arXiv preprint arXiv:2605.27582},
  year={2026}
}
@article{ding2025lavira,
  title={LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision Language Navigation in Continuous Environments},
  author={Ding, Hongyu and Xu, Ziming and Fang, Yudong and Wu, You and Chen, Zixuan and Shi, Jieqi and Huo, Jing and Zhang, Yifan and Gao, Yang},
  journal={arXiv preprint arXiv:2510.19655},
  year={2025}
}
```

## Acknowledgements

This work builds on [LaViRA](https://github.com/NJU-R-L-Group-Embodied-Lab/lavira-code), [CA-Nav](https://github.com/Chenkehan21/CA-Nav-code), [NavDP](https://github.com/InternRobotics/NavDP), and [Depth-Anything V2](https://github.com/DepthAnything/Depth-Anything-V2). Thanks for their great work!

## License

The open-sourced code is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](http://creativecommons.org/licenses/by-nc-sa/4.0/) — see [`LICENSE`](LICENSE). Third-party submodules and vendored components remain under their own upstream licenses.
