# Uni-LaViRA: Language-Vision-Robot Actions Translation for Unified Embodied Navigation

Habitat-based simulation for **Uni-LaViRA**, a training-free framework that *translates* navigation **language → vision → robot action** with pretrained multimodal LLMs. It reproduces the indoor benchmarks — VLN-CE (R2R/RxR), HM3D-v2, HM3D-OVON, and MP3D-EQA — where Uni-LaViRA rivals trained SOTA foundation models with zero training.

## Environment Setup

### 1. Create Conda Environment
Tested with Python 3.8.
```shell
conda create -n lavira python=3.8
conda activate lavira
```

### 2. Install Habitat-Sim & Habitat-Lab
**Habitat-Sim** (compiled locally):
```shell
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim && git checkout tags/v0.1.7
pip install -r requirements.txt
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" python setup.py install --headless --with-cuda --bullet
```
**Habitat-Lab** — first remove `tensorflow==1.13.1` from `habitat_baselines/rl/requirements.txt` to avoid conflicts:
```shell
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab && git checkout tags/v0.1.7
vi habitat_baselines/rl/requirements.txt   # remove the line: tensorflow==1.13.1
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop --all
```

### 3. GroundingDINO
Used to build semantic maps for action planning.
```shell
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
pip install -q -e . --no-build-isolation
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
pip install nltk
```
**Phrase-to-Class mapping** — following [CA-Nav](https://github.com/Chenkehan21/CA-Nav-code), replace `phrases2classes` (around line 235 of `GroundingDINO/groundingdino/util/inference.py`) with an edit-distance version:
```python
from nltk.metrics import edit_distance

@staticmethod
def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
    class_ids = []
    for phrase in phrases:
        if phrase in classes:
            class_ids.append(classes.index(phrase))
        else:
            distances = np.array([edit_distance(phrase, c) for c in classes])
            class_ids.append(np.argmin(distances))
    return np.array(class_ids)
```

### 4. Other dependencies
```shell
pip install setuptools==58.5.3 meson-python ninja
pip install -r requirements.txt --use-pep517 --no-build-isolation
pip install --no-deps -e ../..  # shared LaViRA core, from sim-code/habitat/
```

### 5. NavDP planner
Every task uses [NavDP](https://github.com/InternRobotics/NavDP) for local planning on stairs. Install its one extra dependency:
```shell
pip install -r navdp/requirements.txt
```

## Dataset Preparation

Place everything under `data/` (full layout under [Pretrained Weights](#pretrained-weights)).

**Matterport3D scenes (R2R, RxR, EQA).** Request access on the [Matterport3D page](https://niessner.github.io/Matterport/) to obtain `download_mp.py`, then download the Habitat scenes into `data/scene_datasets/mp3d/`:
```shell
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

**HM3D-Semantics v0.2 scenes (ObjectNav, OVON).** Only the `val` split is used. Request a Matterport API token on the [HM3D dataset page](https://aihabitat.org/datasets/hm3d/), then download it into `data/scene_datasets/hm3d/`:
```shell
python -m habitat_sim.utils.datasets_download \
  --username <TOKEN_ID> --password <TOKEN_SECRET> \
  --uids hm3d_val_v0.2 --data-path data/scene_datasets/hm3d
```

**Episodes (all five tasks).** The `val_unseen` episode files (renamed per task, with NDTW ground truth for R2R/RxR) are bundled on [Google Drive](https://drive.google.com/file/d/1Seiq-2cYVZAb7xX569Mn_PZCDjujNzQa/view) (~33 MB):
```shell
cd data/datasets
gdown https://drive.google.com/uc?id=1Seiq-2cYVZAb7xX569Mn_PZCDjujNzQa
unzip uni_lavira_val_unseen_episodes.zip   # -> data/datasets/episodes/
cd ../..
```
The 100-episode subsets that reproduce the headline numbers ship with the code under `data/datasets/stratified_samples/`; each script filters the full split to these via `--episode-file`.

## Pretrained Weights

Download the GroundedSAM weights from [here](https://drive.google.com/drive/folders/1RvB3z8wi19saplpFYw07NwTdgVkBbH2G) into `data/grounded_sam/`, and the NavDP checkpoint from the [NavDP weights release](https://github.com/InternRobotics/NavDP#-internvla-n1-system-1-model) to `navdp/navdp-cross-modal.ckpt` (the `sim-code/habitat/` directory, **not** under `data/`).

GroundingDINO's text encoder also needs `bert-base-uncased` locally (the eval scripts run offline with `HF_HUB_OFFLINE=1`). Download it from [Hugging Face](https://huggingface.co/google-bert/bert-base-uncased) into `data/grounded_sam/bert-base-uncased/`:

```shell
cd data/grounded_sam && git lfs install && git clone https://huggingface.co/google-bert/bert-base-uncased && cd ../..
```

The eval scripts read it via `BERT_LOCAL_PATH=data/grounded_sam/bert-base-uncased`. Once everything is downloaded, the `data/` directory should be organized as follows:

```shell
data
├── grounded_sam
│   ├── groundingdino_swint_ogc.pth
│   ├── GroundingDINO_SwinT_OGC.py
│   ├── repvit_sam.pt
│   ├── sam_vit_h_4b8939.pth
│   └── bert-base-uncased/        # GroundingDINO text encoder
├── datasets
│   ├── stratified_samples        # ships with the code (100-ep ID lists)
│   └── episodes                  # from the Google Drive bundle
└── scene_datasets
    ├── mp3d                       # 90 MP3D scenes
    │   ├── 17DRP5sb8fy/           #   each scene: .glb / .house / .navmesh / _semantic.ply
    │   └── ...
    └── hm3d                       # HM3D-Sem v0.2 scenes
        └── hm3d_v0.2/
            └── val/               # 100 val scenes
                ├── 00800-TEEsavR23oF/   #   each: .basis.glb / .basis.navmesh / .semantic.glb / .semantic.txt
                └── ...
```

## Evaluation

Set six environment variables (see `.env.example`): `LA_API_KEY`, `LA_BASE_URL`, `LA_MODEL_NAME` for Language Action model and `VA_API_KEY`, `VA_BASE_URL`, `VA_MODEL_NAME` for Vision Action model.

**Recommended:** LA = `gemini-3.5-flash` (or `gemini-3.1-pro`); VA = `qwen3.5-27b`. For a low-cost testing, LA can also be `qwen3.5-27b` (with thinking budget ~1024).

Each script sets `CUDA_VISIBLE_DEVICES=0`, reads worker count from `NPROC` (default 20), and defaults to the 100-episode stratified subset:
```shell
bash eval_scripts/vlnce_r2r.sh    # VLN-CE R2R
bash eval_scripts/vlnce_rxr.sh    # VLN-CE RxR
bash eval_scripts/hm3d_v2.sh      # HM3D-v2
bash eval_scripts/hm3d_ovon.sh    # HM3D-OVON
bash eval_scripts/mp3d_eqa.sh     # MP3D-EQA
```

Each 100-episode task takes roughly 1 hour with 20 parallel workers (`NPROC=20`).

## Docker Setup

Alternatively, build the full environment from the provided `Dockerfile` — Ubuntu 22.04 / CUDA 11.8, Habitat-Sim 0.1.7 (built from source), Habitat-Lab, GroundingDINO, Segment Anything, and a Python 3.8 `lavira` conda env — instead of the manual setup above:

```shell
docker build -t lavira-oss:dev .
```

Download all of `data/` on the host first, then run a container with your GPUs, the code, and the pre-downloaded `data/` mounted in via `-v`:

```shell
docker run -it --gpus all \
  --env-file .env \
  -v "$(pwd)/../..":/workspace/uni-lavira \
  -v path/to/data:/workspace/uni-lavira/sim-code/habitat/data \
  -w /workspace/uni-lavira/sim-code/habitat \
  lavira-oss:dev bash
```

Run these commands from `sim-code/habitat/`. The whole repository is mounted so the shared package is available. Inside the container, run `pip install --no-deps -e ../..` once before the eval scripts. `--env-file .env` passes the six LA/VA variables into the container; `bert-base-uncased` is loaded from the mounted `data/` via `BERT_LOCAL_PATH`.

## Visualization

Runs write per-step frames to `saved_rgb_images/<exp_name>/`. View them with:
```shell
python server.py   # then open http://localhost:9999/?root=saved_rgb_images/<exp_name>
```

## Credits

If you find this work useful, please cite:
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

This work builds on [LaViRA](https://github.com/NJU-R-L-Group-Embodied-Lab/lavira-code), [CA-Nav](https://github.com/Chenkehan21/CA-Nav-code), [NavDP](https://github.com/InternRobotics/NavDP), and [Depth-Anything V2](https://github.com/DepthAnything/Depth-Anything-V2). Thanks for their great work!
