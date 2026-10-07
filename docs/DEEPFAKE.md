# Deepfake target — C-DUP and U3D against SimSwap face swapping

The C3D target measures whether a universal perturbation makes a video **classifier** fail. This
target asks the question the perturbations are ultimately meant for: if a user adds the
perturbation to a video **before uploading it**, does a face-swapping model fail to turn that
video into a deepfake?

The model under attack is **SimSwap**, and the defence goal and judges are taken unmodified from
**PhantomSeal** (Ren et al., CCS 2026), a per-image proactive deepfake defence. What changes is
how the perturbation is made: instead of optimising every image, one perturbation clip is built
with the C-DUP or U3D machinery — either once for all videos (universal) or once per video.

---

## 1. What is reused and what is replaced

| | Reused unchanged | Replaced |
|---|---|---|
| **C-DUP** | 3D generator, 2D generator + Tile, Roll, Adam schedule, drawing one perturbation (`common/dfbench/cdup_core.py`, shared with the C3D target) | loss: C3D cross-entropy → PhantomSeal protection score. C-DUP's "dual purpose" term has no counterpart: every face must be protected |
| **U3D** | Perlin noise, temporal shift Trans(ξ, τ) with I = 5, upstream Rust PSO, search bounds (`common/dfbench/u3d_perlin.py`, shared) | objective: C3D feature distance → PhantomSeal protection score |
| **PhantomSeal** | SimSwap 224, ArcFace identity extractor, netG encoder (context), the identity / context deviations and their caps, FaceNet-512 and dlib judges | the per-image utility term → a hard L∞ budget (a universal perturbation has fixed amplitude) |

The protection score is PhantomSeal's two deviations, each divided by PhantomSeal's own cap and
averaged, so it lies in [0, 1]:

```
score = ( min(D_id, 0.003)/0.003 + min(D_ctx, 25)/25 ) / 2
D_id  = mean((id(x~) - id(x))^2)     ArcFace identity embedding
D_ctx = mean((E(x~)  - E(x))^2)      SimSwap encoder features
```

The decoy (cloak) term of PhantomSeal is not used.

## 2. Pipeline

```
 run    train_universal.py (train faces, once)   per_video.py (each test video)
        evaluate.py (SimSwap re-detection attack + PhantomSeal judges)
 ─────────────────────────────────────────────────────────────────────────────
 method u3d_core.py   Perlin noise (paper Eq. 1-2), temporal shift, Rust PSO
        dfbench/cdup_core.py   3D / 2D generator, Roll, Adam
        optimize.py   u3d_search / cdup_train, driven by the protection score
 ─────────────────────────────────────────────────────────────────────────────
 bridge inject.py, common.py   clip 16x112x112 -> 224 face-crop space -> placed on the
                               frame -> cropped again the way the attacker does (differentiable)
 ─────────────────────────────────────────────────────────────────────────────
 target model.py   SimSwap + ArcFace + encoder, protection score   (PhantomSeal, unmodified)
        judges.py  FaceNet-512 / dlib, identical to PhantomSeal's src/evaluate.py
```

**Face-anchored injection.** U3D and C-DUP add their clip to the C3D input (16 × 112 × 112).
SimSwap's input is the 224 × 224 aligned face crop, so the clip is defined in that space: it is
upsampled 112 → 224 bilinearly, tapered to zero over the last 16 px of the crop border, and
warped onto the frame with the inverse of the defender's face alignment. Frame t receives clip
frame (t + τ) mod 16. The defender only runs a face detector; nothing is optimised per video in
the universal setting. The upload is uint8 and every channel stays within ±10 levels.

**Three ways to protect a video.** The comparison spans how much optimisation happens per input:

| | Perturbation | Optimised on | Cost at protection time |
|---|---|---|---|
| universal (UAP) | one C-DUP / U3D clip for every video | UCF-101 train faces, once | face detection only |
| per-video | one C-DUP / U3D clip per video | that video's frames | one optimisation per video |
| per-frame | PhantomSeal on every frame | each frame on its own | one optimisation per frame |

The per-frame reference (`phantomseal_frames.py`) runs PhantomSeal's own objective — identity,
decoy (cloak), context and utility terms, sign-gradient, 1,000 iterations — on each scored frame.
It optimises the frame pixels through the defender's crop warp: optimising the 224 crop and
pasting it back loses most of the perturbation once the attacker crops again. Its per-channel
bound is the same 10/255 as the universal rows; `--native-limits` uses PhantomSeal's own R/G/B
bounds (0.075 / 0.030 / 0.075) for reference.

**Attacker.** Evaluation runs SimSwap's real video pipeline on the protected upload: it
re-detects the face, aligns it, crops 224 × 224 and swaps. `--oracle` additionally scores the
paper's protocol, where the attacker reuses the defender's alignment.

- *Identity theft*: the protected face is swapped onto another person; it succeeds if the result is
  recognised as the victim.
- *Context theft*: another person's face is swapped into the protected video; it succeeds if the
  result is recognised as that person.

## 3. U3D: the corrected search is the default

Two defects of the U3D reference implementation (alarst13/u3d) are corrected here by default:

| | Reference implementation | Default here |
|---|---|---|
| Noise | Rust colour map divides, `sin(v·2π/color_period)`, on unipolar octaves: most of the parameter space collapses into a near-uniform brightness shift | paper Eq. 1–2: `sin(N·2πφ)`, bipolar octaves, Λ+1 terms |
| PSO inertia | psolib updates `v ← v + ω·v + …`; ω = 1.2 is an effective inertia of 2.2, the swarm diverges and every particle ends on the bounds | ω = −0.27, effective inertia 0.73; the swarm settles inside the box |

Earlier runs can be reproduced for comparison and are evaluated under their own variant names:

```bash
python -m deepfake.train_universal --method u3d --n-val 150 --u3d-noise repo --u3d-omega 1.2 --tag _repo
python -m deepfake.per_video --method u3d --u3d-noise repo --u3d-omega 1.2 --tag _repo
python -m deepfake.evaluate --name main --oracle --variants u3d_repo_uap,u3d_repo_pv
```

## 4. Installation

The deepfake target runs in PhantomSeal's environment (Python 3.10.19, PyTorch 2.8, CUDA 12.8).
The C3D target also runs there, with results identical to its own lighter environment, so one
environment serves both.

```bash
git clone https://github.com/LiangqinRen/PhantomSeal.git repos/PhantomSeal
cd repos/PhantomSeal
bash tools/create_env.sh             # creates the "phantomseal" environment
bash tools/setup.sh                  # third-party projects, patches, datasets and checkpoints (~28 GB)
cd ../..

conda activate phantomseal
pip install decord                   # the C3D target's video decoder
bash scripts/setup_u3d.sh            # U3D's Rust extensions (perlin, psolib)
PYTHONPATH=common python scripts/check_env.py
python -m deepfake.selftest
```

UCF-101 comes from the C3D target's preparation step (`scripts/00_prepare_dataset.sh`); both
targets read the same `dataset/`.

> Deactivate any other virtual environment first. With `VIRTUAL_ENV` set, `conda run` and `pip`
> can install into that environment instead, and `maturin` refuses to build.
>
> `tools/create_env.sh` installs `facenet-pytorch` with `--no-deps` on purpose: its metadata pins
> `torch < 2.3`, which has no kernels for RTX 50-series GPUs.

| Variable | Default | Meaning |
|---|---|---|
| `PHANTOMSEAL_ROOT` | `repos/PhantomSeal` | PhantomSeal checkout (code, third-party models, checkpoints) |
| `DEEPFAKE_CACHE` | `cache/deepfake` | face scans and decoded train face frames (~3 GB) |
| `DEEPFAKE_RESULTS` | `results/deepfake` | all outputs of this target |

## 5. Running

```bash
bash scripts/run_deepfake.sh                 # every stage, in order
STAGES="5 6" bash scripts/run_deepfake.sh    # selected stages
```

| Stage | Command | Output |
|---|---|---|
| 0 | `python -m deepfake.selftest` | 14 checks: ports equal the originals, the PSO converges |
| 1 | `python -m deepfake.scan_faces --split test` / `--split train --save-frames` | face scans, train face frames (cache) |
| 2 | `python -m deepfake.train_universal --method {u3d,cdup3d,cdup2d,uniform} --n-val 150` | `uap/universal/<method>/clip.npy`, `meta.json` |
| 3 | `python -m deepfake.prepare_eval --n 100` | `eval_set.json`: 100 test videos, attacker partners |
| 4 | `python -m deepfake.per_video --method {u3d,cdup3d}` | `uap/per_video/<method>/<video>.npy` |
| 4b | `python -m deepfake.phantomseal_frames` | `uap/per_frame/phantomseal/<video>.npz` (protected scored frames) |
| 5 | `python -m deepfake.evaluate --name main --oracle` | `eval/main/records.jsonl` (one line per video × variant) |
| 6 | `python -m deepfake.summarize --name main --oracle` | result tables |
| 7 | `python -m deepfake.render_demo` | side-by-side videos: protected upload, noise, both deepfakes |

Universal perturbations are built from UCF-101 split-1 **train** faces and evaluated on **test**
videos, whose groups do not overlap with train. `evaluate.py --codec xvid` re-encodes the
protected upload before the attack.

## 6. Reproducibility

- SimSwap, ArcFace and FaceNet run on the GPU with non-deterministic kernels. Evaluating the same
  perturbation twice flips under 1% of the per-frame FaceNet decisions.
- The U3D search is not bit-reproducible: psolib's Rust PSO uses an unseeded random generator.
- A drawn C-DUP perturbation is reproducible from its generator and seed (`draw_perturbation`).

## 7. Attribution

PhantomSeal is MIT-licensed (Copyright © 2026 Liangqin Ren). SimSwap, ArcFace, FaceNet, dlib and the
other third-party models are downloaded by PhantomSeal's setup and remain under their own
licenses. See [`NOTICE`](../NOTICE).
