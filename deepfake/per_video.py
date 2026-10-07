"""Step 4 - the per-video alternative: optimise a perturbation for each TEST video on its own frames.

Same U3D / C-DUP modules and the same PhantomSeal objective as the universal version; the only
difference is the data: the faces of the evaluation window [0, 80) of the video being protected.

    python -m deepfake.per_video --method u3d      # PSO per video (16 faces x I=5 shifts)
    python -m deepfake.per_video --method cdup3d   # generator trained from scratch per video

Output: results/uap/per_video/<method>/<stem>.npy ([16, C, 112, 112]) + meta.jsonl (resumable).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from deepfake.common import EPS, FaceBatch, seed_everything
from deepfake.model import ProtectionObjective, SimSwapTarget
from deepfake import u3d_core
from deepfake.optimize import cdup_clip, cdup_train, u3d_clip, u3d_search
from deepfake.paths import CACHE, RESULTS
from deepfake.ucf import read_frames


def video_faces(entry):
    rec = json.load(open(CACHE / "dets" / f"{Path(entry['video']).stem}.json"))
    frames, _ = read_frames(entry["video"], rec["n_window"])
    ts = [t for t, d in enumerate(rec["dets"]) if d is not None]
    return frames, rec["dets"], ts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["u3d", "cdup3d"], required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n-fit", type=int, default=16, help="U3D: faces per video in the fitness")
    p.add_argument("--eps", type=float, default=EPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", default="0/1", help="k/n: process every n-th video starting at k")
    p.add_argument("--eval-set", default="eval_set.json")
    p.add_argument("--only", default="", help="comma-separated video stems (default: all)")
    p.add_argument("--tag", default="", help="output directory suffix, e.g. _repo")
    p.add_argument("--u3d-noise", choices=["paper", "repo"], default=u3d_core.NOISE)
    p.add_argument("--u3d-omega", type=float, default=u3d_core.OMEGA)
    a = p.parse_args()

    out = RESULTS / "uap" / "per_video" / (a.method + a.tag)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / f"run_{a.shard.replace('/', 'of')}.log"),
                                  logging.StreamHandler()])
    log = logging.getLogger("deepfake").info
    target = SimSwapTarget()
    objective = ProtectionObjective(target)
    entries = json.load(open(RESULTS / a.eval_set))["videos"]
    if a.only:
        keep = set(a.only.split(","))
        entries = [e for e in entries if Path(e["video"]).stem in keep]
    k, n = map(int, a.shard.split("/"))
    entries = entries[k::n]

    for i, e in enumerate(entries):
        stem = Path(e["video"]).stem
        if (out / f"{stem}.npy").exists():
            continue
        seed_everything(a.seed)
        t0 = time.time()
        frames, dets, ts = video_faces(e)
        fb = FaceBatch.build([frames[t] for t in ts], [np.asarray(dets[t]["M"]) for t in ts], objective)
        if a.method == "u3d":
            pick = np.linspace(0, len(fb) - 1, min(a.n_fit, len(fb))).round().astype(int)
            best, fit, hist = u3d_search(objective, fb.subset(torch.as_tensor(pick, device="cuda")),
                                         seed=a.seed, eps=a.eps, log=None,
                                         noise=a.u3d_noise, omega=a.u3d_omega)
            clip = u3d_clip(best, a.eps, a.u3d_noise)
            extra = {"params": best, "fit_score": fit, "evals": len(hist),
                     "noise": a.u3d_noise, "omega": a.u3d_omega}
        else:
            rng = np.random.RandomState(a.seed)

            def sample_batch(m):
                return fb.subset(torch.as_tensor(rng.randint(0, len(fb), size=m), device="cuda"))

            g, hist = cdup_train(objective, sample_batch, steps=a.steps, batch=a.batch, eps=a.eps,
                                 seed=a.seed, log=None)
            clip = cdup_clip(g, a.eps, seed=a.seed)
            extra = {"steps": a.steps, "final_train_score": hist[-1]["score"], "history": hist}
        np.save(out / f"{stem}.npy", clip.detach().cpu().numpy().astype(np.float32))
        sec = round(time.time() - t0, 1)
        with open(out / "meta.jsonl", "a") as f:
            f.write(json.dumps({"video": e["video"], "n_faces": len(fb), "seconds": sec, **extra}) + "\n")
        log(f"[{a.method} per-video] {i+1}/{len(entries)} {stem}: {sec}s "
            f"{'fit' if a.method == 'u3d' else 'train'} score "
            f"{extra.get('fit_score', extra.get('final_train_score')):.4f}")
        del fb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
