# Unitree Go1 — Real-World Multi-Task Navigation for Uni-LaViRA

The training-free **Uni-LaViRA** stack — pretrained LLMs *translated* through **language → vision → robot action** — runs **VLN**, **ObjectNav**, and **EQA** on the Unitree Go1 quadruped with no fine-tuning; only the low-level controller is platform-specific.

## Tasks

| `--task` | Goal |
|----------|------|
| `vln` | Follow a natural-language route instruction. |
| `object_nav` | Reach a named object (text goal, e.g. `"chair"`). |
| `eqa` | Navigate, then answer a question about the scene. |
| `interact` | Web + voice demo (instruction given at runtime). |

All tasks share one pipeline and one LA/VA model; only the instruction differs.

## Pipeline

Per cycle: capture a 4-direction panorama (front / back / left / right) from the four head cameras → **LA** updates a markdown TODO list and picks the next heading (or `stop`) → base turns to face it → **VA** returns `STOP` or the next waypoint's bounding box → the bbox is back-projected through front-camera depth to a robot-frame goal, which iPlanner drives with pure-pursuit and parallel replanning.

## Layout

```
unitree_go1/
├── main.py            # Entry; --task {vln,object_nav,eqa,interact}
├── config.py          # Env-var overridable configuration
├── prompts.py
├── utils.py
├── ai_client/         # LaViRAVisionClient (two-endpoint LA + VA)
├── robot/             # RobotController, iplanner_client, navigation_api, nav_controller
├── tasks/             # vln, object_nav, eqa
├── iplanner/          # iPlanner server + model
└── web/               # Flask + SocketIO voice/text demo
```

## Install

```bash
conda create -n unitree_go1 python=3.10 && conda activate unitree_go1
pip install -r requirements.txt
pip install -e ../..                     # shared LaViRA core, from this directory
```

Also required:
- ROS Noetic with the four Orbbec Gemini 336L cameras publishing `/cameraN/color|depth/image_raw` (`N = 1..4`).
- Unitree High-level SDK (`robot_interface.so`) — set `UNITREE_SDK_PATH`, or put it on `PYTHONPATH`.
- iPlanner checkpoint: place `iplanner.pth` at `iplanner/checkpoints/iplanner.pth`.

## Models (llama.cpp)

One local `Qwen3.5-27B-Q4_K_M` GGUF serves both LA and VA via `llama-server` (OpenAI-compatible `/v1`):

```bash
llama-server --model path/to/Qwen3.5-27B-Q4_K_M.gguf --mmproj path/to/mmproj.gguf \
    --alias Qwen3.5-27B-Q4_K_M --host 0.0.0.0 --port 8000 --ctx-size 8192 --n-gpu-layers 999
```

`--mmproj` (vision projector) is required for both LA and VA calls. For a remote API, set the `LA_*` / `VA_*` variables instead.

## Run

```bash
roscore
roslaunch your_go1_bringup.launch        # cameras + UDP bridge
python iplanner/iplanner_server.py       # iPlanner server (port 8888)

python main.py --task vln        --instruction "go to the chair, then turn right"
python main.py --task object_nav --instruction "chair"
python main.py --task eqa        --instruction "what colour is the sofa?"
python main.py --task interact            # web + voice demo at https://<robot-ip>:5000
```

The web demo needs a self-signed `cert.pem`/`key.pem`; voice needs a local faster-whisper model (optional — text always works).

## Environment variables

Key env vars (all overridable; defaults in `config.py`):

| Variable | Default |
|----------|---------|
| `LA_BASE_URL` / `LA_MODEL_NAME` | `http://localhost:8000/v1` / `Qwen3.5-27B-Q4_K_M` |
| `VA_BASE_URL` / `VA_MODEL_NAME` | same as LA (single local model) |
| `IPLANNER_URL` | `http://localhost:8888` |
| `UNITREE_SDK_PATH` / `UNITREE_HOST` | `""` / `192.168.123.161` |
| `COBOT_HTTP_PORT` / `FLASK_SECRET_KEY` | `5000` / `change-me` |

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
