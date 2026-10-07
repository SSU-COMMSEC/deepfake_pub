# Universal Adversarial Perturbation (UAP) Baselines for Video — C-DUP & U3D

Faithful reproductions of two **universal adversarial perturbation (UAP)** attacks on video
recognition models, evaluated under a single common protocol — one victim model, one split,
one perturbation budget, one set of metrics.

| Method | Venue | Perturbation | Optimized variable | Threat model | Docs |
|---|---|---|---|---|---|
| **C-DUP** | NDSS 2019 | Dense tensor produced by a generator | Generator weights (gradient descent) | White-box | [docs/CDUP.md](docs/CDUP.md) |
| **U3D** | IEEE S&P 2022 | Procedural Perlin noise | Five parameters (PSO) | Transfer-based black-box | [docs/U3D.md](docs/U3D.md) |

Both methods run against two targets that share the same perturbation cores (`common/dfbench/`):

| Target | Model under attack | Question | Code | Docs |
|---|---|---|---|---|
| **C3D** | C3D video classifier | does the perturbation make the video be misclassified? | `adapters/` | this README |
| **Deepfake** | SimSwap face swapping, PhantomSeal's defence objective and judges | does a video protected before upload resist being turned into a deepfake? | `deepfake/` | [docs/DEEPFAKE.md](docs/DEEPFAKE.md) |

---

## Quick start

Both perturbations ship with the repository, so the attacks can be seen **without training a
generator, running a search, or downloading the dataset.** A single video file is enough.

```bash
conda env create -f environment.yml
conda activate uap-baselines
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your driver

PYTHONPATH=common python scripts/demo.py --video assets/demo.avi --method both
```

```
clean   : ApplyEyeMakeup  (64.9 %)
[cdup] L_inf 10.000  mean|delta| 8.942  ->  Hammering     (67.4 %)  -- ATTACK SUCCEEDS
[u3d]  L_inf 10.000  mean|delta| 5.762  ->  PlayingTabla  (81.2 %)  -- ATTACK SUCCEEDS
```

Each run writes a three-panel **original | protected | perturbation (×8)** video to
`demo_out/`. The prediction lines appear only when the victim checkpoint is present (step 1 of
[Installation](#installation)); the video is rendered either way.

---

## Evaluation protocol

| | |
|---|---|
| Dataset | UCF-101, official split-1 |
| Victim model | C3D (trained on UCF-101) |
| Perturbation budget | L∞ ≤ 10 / 255 |
| Evaluation set | 101 videos, one per class, all correctly classified |
| Success criterion | Misclassification **and** `‖δ‖∞ ≤ ε`, judged by the shared victim model |

Each run records the fooling rate, mean query count and mean absolute perturbation in
`results/<name>/summary.json`.

---

## Repository structure

```
uap-baselines/
├── NOTICE                        attribution for the papers and reference implementations
├── environment.yml               conda environment definition
├── requirements.txt              pip-only installation path
│
├── docs/
│   ├── CDUP.md                   C-DUP: method, code map, deviations, usage
│   ├── U3D.md                    U3D:   method, code map, findings, usage
│   └── DEEPFAKE.md               deepfake target: integration, corrected U3D, installation, usage
│
├── assets/
│   ├── figures/                  architecture figures from the papers (authors' work, attributed)
│   └── perturbations/            pre-generated UAPs (shipped with the repository)
│       ├── cdup_perturbation.npy (3, 16, 112, 112) float32
│       ├── u3d_perturbation.npy  (16, 112, 112)    float64
│       ├── u3d_params.json       the five parameters that regenerate the U3D noise
│       ├── SHA256SUMS
│       └── README.md             provenance and shapes
│
├── adapters/                     C3D target: end-to-end runner per method — generate -> save -> evaluate
│   ├── cdup_adapter.py           trains the generator (dfbench/cdup_core.py) against C3D and
│   │                             extracts one perturbation
│   └── u3d_adapter.py            searches the five parameters with PSO and synthesizes the noise;
│                                 imports the reference implementation in repos/u3d/ unmodified
│
├── deepfake/                     deepfake target (SimSwap + PhantomSeal objective); see docs/DEEPFAKE.md
│   ├── u3d_core.py, optimize.py  U3D (paper noise, converging PSO) and C-DUP driven by the protection score
│   ├── inject.py, common.py      clip -> face-crop space -> frame -> attacker crop (differentiable)
│   ├── model.py, judges.py       PhantomSeal's SimSwap target, objective, FaceNet-512 / dlib judges
│   ├── phantomseal_frames.py     PhantomSeal on every frame (per-frame reference)
│   └── scan_faces.py ... render_demo.py   stages 1-7, selftest.py
│
├── common/
│   ├── setup.py                  makes `dfbench` installable (`pip install -e common`)
│   └── dfbench/                  shared harness and perturbation cores — used by both methods and both targets
│       ├── cdup_core.py          C-DUP generators (3D, 2D), Roll, Adam schedule, drawing one perturbation
│       ├── paths.py              single source of every path; falls back to the repo root without DF_ROOT
│       ├── data.py               decoding and preprocessing identical to the victim's training pipeline
│       │                         (center 16 frames, resize, 112x112 crop, mean subtraction)
│       ├── victim.py             the attacked C3D; a standalone re-implementation without mmcv that
│       │                         counts its own queries and exposes intermediate activations for U3D
│       ├── runner.py             evaluation loop: iterate the frozen subset -> call the attack ->
│       │                         judge success -> one record per video -> skip finished videos
│       ├── metrics.py            aggregates fooling rate, mean queries and mean absolute perturbation
│       ├── records.py            crash-safe storage (immediate append + atomic writes)
│       ├── u3d_perlin.py         U3D noise, paper Eq. 1-2 and the Rust-compatible variant (numpy and GPU)
│       └── trt_victim.py         TensorRT-engine victim model (optional)
│
└── scripts/
    ├── check_env.py              installation check — prints what is missing and the command to fix it
    ├── demo.py                   applies a perturbation to a video and renders a three-panel comparison
    ├── 00_prepare_dataset.sh     places and verifies UCF-101, the split lists and the C3D checkpoint
    ├── 01_clean_eval.py          measures clean accuracy and freezes the evaluation set
    ├── build_train_cache.py      decodes the training clips once into a memory map
    ├── setup_u3d.sh              clones alarst13/u3d and builds the Rust extensions
    └── run_deepfake.sh           deepfake target, stages 0-7

                                  created during installation and not tracked:
                                  dataset/  checkpoints/  repos/  results/  demo_out/  cache/
```

## Development environment

| Item | Value | | Package | Verified version | Minimum |
|---|---|---|---|---|---|
| OS | Ubuntu 22.04.5 LTS (kernel 6.8) | | Python | 3.10.21 | 3.10 |
| GPU | RTX 5090 32 GB · sm_120 | | PyTorch | 2.7.1+cu128 | **2.7** |
| Driver · CUDA | 595.91.07 · 12.8 | | NumPy | 1.26.4 / 2.2.6 | 1.24 |
| CPU | Core i7-14700K (28 threads) | | OpenCV | 4.11.0 | 4.8 (5.x unsupported) |
| RAM | 31 GB | | decord | 0.6.0 | 0.6 (optional) |
| conda | 26.1.1 | | Rust · maturin | 1.99.0 · 1.15.0 | 1.70 · 1.0 (U3D only) |
| | | | TensorRT | 10.16.0.72 | — (optional, `trt_victim.py`) |

| Resource | Requirement |
|---|---|
| GPU memory | **8 GB or more** recommended for generator training and the PSO search. Raising U3D's `--opt_batch` (default 8) substantially risks running out of memory |
| Disk | Dataset about 7 GB (13 GB if the 6.5 GB `UCF101.rar` is kept), checkpoint 313 MB, training-clip cache 5.74 GB → reserve **about 20 GB** |
| CPU | `01_clean_eval.py` decodes 3,783 videos, so setting `--workers` to the core count speeds it up considerably |

For the demo alone, a GPU is optional (`--device cpu`) and a single video is all the disk needed.

---

## Installation

```bash
conda env create -f environment.yml
conda activate uap-baselines

# PyTorch is kept out of the environment definition because its CUDA build must match your driver. Always run this as well
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8

pip install -r requirements.txt
```

This environment covers the C3D target. The deepfake target runs in PhantomSeal's environment
(PyTorch 2.8), which also runs the C3D target with identical results — one environment for both.
See [docs/DEEPFAKE.md §4](docs/DEEPFAKE.md#4-installation).

### 1. Dataset and victim checkpoint
| Asset | Download | Size |
|---|---|---|
| UCF-101 videos | [`UCF101.rar`](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) | about 6.5 GB |
| Official split lists | [`UCF101TrainTestSplits-RecognitionTask.zip`](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip) | about 114 KB |
| C3D victim checkpoint | [`c3d_sports1m_...-26655025.pth`](https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth) | about 313 MB |

Place the downloaded files **without extracting them** at the locations below, then run the
preparation script.

```bash
mkdir -p dataset checkpoints
mv ~/Downloads/UCF101.rar                                  dataset/
mv ~/Downloads/UCF101TrainTestSplits-RecognitionTask.zip   dataset/
mv ~/Downloads/c3d_sports1m_*.pth                          checkpoints/

bash scripts/00_prepare_dataset.sh
```

### 2. Preprocessing

```bash
# Measure clean accuracy and freeze the evaluation set from correctly classified videos only.
PYTHONPATH=common python scripts/01_clean_eval.py --workers 12 --per-class 3

# Pre-decode the 9,537 training videos into a memory-mapped cache (5.74 GB, resumable).
PYTHONPATH=common python scripts/build_train_cache.py

# U3D only: clone the reference implementation and build the perlin and psolib Rust extensions.
# Requires the Rust toolchain (https://rustup.rs).
bash scripts/setup_u3d.sh
```

---

## Running

```bash
PYTHONPATH=common python adapters/cdup_adapter.py --tier 1     # C-DUP
PYTHONPATH=common python adapters/u3d_adapter.py  --tier 1     # U3D
```

Each command runs the training or search, saves the perturbation, and applies it to the
frozen evaluation set to score every video. **This is a full reproduction run, not a demo.**

**Both runs are resumable.** The C-DUP generator checkpoint (`generator.pt`) and the U3D PSO
result (`u3d_params.json`) are cached and reused, and finished videos are skipped. Running the
same command again after a complete run therefore **finishes in a few seconds, without
re-optimizing or re-evaluating.** Pass `--force_train` / `--force_optimise` to recompute, or
delete `results/<method>/records.jsonl` to re-score.

### Runtime
| Stage | Time | Notes |
|---|---|---|
| `01_clean_eval.py --workers 12` | **2.9 min** | decode + inference over 3,779 videos |
| `build_train_cache.py` | **about 3 min** | 9,537 videos → 5.74 GB memory map |
| C-DUP generator training | **1.7 min** | 447 steps (3 epochs × 149, batch 64) |
| C-DUP evaluation | **2 s** | 101 videos, 0.019 s per video |
| U3D PSO search | **17.8 min** | 820 objective evaluations (20 particles × 41 generations) |
| U3D evaluation | **2 s** | 101 videos |


### Outputs

| Path | Contents |
|---|---|
| `results/cdup/generator.pt` | trained C-DUP generator |
| `results/cdup/cdup_perturbation.npy` | C-DUP UAP, `(3, 16, 112, 112)` float32 |
| `results/u3d/u3d_params.json` | the five U3D parameters and the search configuration |
| `results/u3d/u3d_perturbation.npy` | U3D UAP, `(16, 112, 112)` float32 |
| `results/<name>/records.jsonl` | one line per video — success, query count, L∞, MAP |
| `results/<name>/summary.json` | aggregate metrics for the run |


## License and attribution

Copyright in the original papers and reference implementations belongs to their respective
authors. See [`NOTICE`](NOTICE) for full attribution. The U3D reference implementation is not
included in this repository; `scripts/setup_u3d.sh` downloads it under its own MIT license.

```bibtex
@inproceedings{li2019stealthy,
  title     = {Stealthy Adversarial Perturbations Against Real-Time Video Classification Systems},
  author    = {Li, Shasha and Neupane, Ajaya and Paul, Sujoy and Song, Chengyu and
               Krishnamurthy, Srikanth V. and Roy Chowdhury, Amit K. and Swami, Ananthram},
  booktitle = {Network and Distributed System Security Symposium (NDSS)},
  year      = {2019}
}

@inproceedings{xie2022universal,
  title     = {Universal 3-Dimensional Perturbations for Black-Box Attacks on
               Video Recognition Systems},
  author    = {Xie, Shangyu and Wang, Han and Kong, Yu and Hong, Yuan},
  booktitle = {IEEE Symposium on Security and Privacy (S\&P)},
  pages     = {1390--1407},
  year      = {2022}
}
```
