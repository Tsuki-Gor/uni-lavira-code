# Agilex Cobot Magic — Real-World Multi-Task Navigation for Uni-LaViRA

The training-free **Uni-LaViRA** stack — pretrained LLMs *translated* through **language → vision → robot action** — runs **VLN**, **ObjectNav**, and **EQA** on the Agilex Cobot Magic wheeled bimanual platform with no fine-tuning; only the low-level controller is platform-specific.

## Tasks

| `--task` | Goal |
|----------|------|
| `vln` | Follow a natural-language route instruction. |
| `object_nav` | Reach a named object (text goal, e.g. `"chair"`). |
| `eqa` | Navigate, then answer a question about the scene. |
| `interact` | Web + voice demo (instruction given at runtime). |

All tasks share one pipeline and one LA/VA model; only the instruction differs.

## Pipeline

Per cycle: the arms sweep a fixed joint sequence to capture a 7-view 45° panorama → **LA** updates a markdown TODO list and picks a heading (or `stop`) → base turns to face it → **VA** returns `STOP` or the next waypoint's bounding box → the bbox is back-projected through front-camera depth to a robot-frame goal, which the in-process iPlanner drives with pure-pursuit and continuous replanning.

## Layout

```
cobot_magic/
├── main.py            # Entry; --task {object_nav,vln,eqa,interact}
├── config.py          # Env-var overridable configuration
├── prompts.py
├── utils.py
├── run_llama_server.sh   # Launch the local llama.cpp model server
├── ai_client/         # LaViRAVisionClient (two-endpoint LA + VA)
├── robot/             # RobotController, arm_controller, iplanner_client, navigation_api, nav_controller
├── tasks/             # vln, object_nav, eqa (+ Factory/Registry)
├── iplanner/          # iPlanner / NavDP model (in-process; checkpoint not shipped)
├── web/               # Flask + SocketIO voice/text demo
├── tools/             # make_videos.py
└── tests/
```

## Install

```bash
conda create -n cobot_magic python=3.10 && conda activate cobot_magic
pip install -r requirements.txt
pip install -e ../..                     # shared LaViRA core, from this directory
```

Also required:
- ROS Noetic with the arm drivers (`/puppet/joint_left|right`), front RealSense (`/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`), and side cameras (`/camera_l|_r/color/image_raw`).
- iPlanner checkpoint at `iplanner/checkpoints/iplanner.pth`.

## Models (llama.cpp)

One local `Qwen3.5-27B-Q4_K_M` GGUF serves both LA and VA via `llama-server` (OpenAI-compatible `/v1`):

```bash
llama-server --model path/to/Qwen3.5-27B-Q4_K_M.gguf --mmproj path/to/mmproj.gguf \
    --alias Qwen3.5-27B-Q4_K_M --host 0.0.0.0 --port 8000 --ctx-size 8192 --n-gpu-layers 999
```

`run_llama_server.sh` wraps this exact command — override the paths via env vars:

```bash
MODEL_PATH=path/to/Qwen3.5-27B-Q4_K_M.gguf MMPROJ_PATH=path/to/mmproj.gguf bash run_llama_server.sh
```

`--mmproj` (vision projector) is required for both LA and VA calls. For a remote API, set the `LA_*` / `VA_*` variables instead.

## Run

```bash
roscore
roslaunch your_cobot_magic_bringup.launch   # arm drivers + cameras

python main.py --task object_nav --instruction "chair"
python main.py --task vln        --instruction "go to the door on the left"
python main.py --task eqa        --instruction "what colour is the sofa?"
python main.py --task interact              # web + voice demo at https://<robot-ip>:5000
```

The web demo needs a self-signed `cert.pem`/`key.pem`; voice needs a local faster-whisper model. Encode run frames to video with `python tools/make_videos.py`.

## Environment variables

Key env vars (all overridable; defaults in `config.py`):

| Variable | Default |
|----------|---------|
| `LA_BASE_URL` / `LA_MODEL_NAME` | `http://localhost:8000/v1` / `Qwen3.5-27B-Q4_K_M` |
| `VA_BASE_URL` / `VA_MODEL_NAME` | same as LA (single local model) |
| `IPLANNER_CHECKPOINT` / `IPLANNER_DEVICE` | `iplanner/checkpoints/iplanner.pth` / `cuda:0` |
| `OUTPUT_ROOT` / `FLASK_SECRET_KEY` | `outputs` / `change-me` |

## Citation

```bibtex
@article{ding2026unilavira,
  title   = {Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation},
  author  = {Ding, Hongyu and Zhang, Sizhuo and Xu, Ziming and Guo, Jinwen and Liu, Hongxiu and Cheng, Xingzhi and Chen, Zixuan and Qi, Haifei and Wang, Duo and Xu, Hao and Shi, Jieqi and Zhang, Yifan and Huo, Jing and Cheng, Jian and Gao, Yang and Luo, Jiebo},
  journal = {arXiv preprint arXiv:2605.27582},
  year    = {2026}
}
@article{ding2025lavira,
  title   = {LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision Language Navigation in Continuous Environments},
  author  = {Ding, Hongyu and Xu, Ziming and Fang, Yudong and Wu, You and Chen, Zixuan and Shi, Jieqi and Huo, Jing and Zhang, Yifan and Gao, Yang},
  journal = {arXiv preprint arXiv:2510.19655},
  year    = {2025}
}
```
