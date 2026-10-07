"""Step 2 - build ONE universal perturbation per method from UCF-101 split-1 TRAIN faces.

    python -m deepfake.train_universal --method u3d      # U3D-based  (Perlin + PSO, paper noise)
    python -m deepfake.train_universal --method cdup3d   # C-DUP-based (3D generator + Roll)
    python -m deepfake.train_universal --method cdup2d   # C-DUP's 2D-DUP (2D generator + Tile)
    python -m deepfake.train_universal --method uniform  # random-noise baseline

Output: results/uap/universal/<method>/{clip.npy, meta.json[, generator.pt]}
clip.npy is [16, C, 112, 112] float32 in [-eps, eps] (C = 1 for U3D, 3 otherwise).
A held-out set of train faces (other videos) gives a quick validation score.
"""

import argparse
import logging
import time

import numpy as np
import torch

from deepfake.common import (EPS, FaceBatch, dump_json, load_train_index, one_frame_per_video,
                        seed_everything)
from deepfake.model import ProtectionObjective, SimSwapTarget
from deepfake import u3d_core
from deepfake.optimize import cdup_clip, cdup_train, clip_score, u3d_clip, u3d_search, uniform_clip
from deepfake.paths import RESULTS


def batch_from(raw, recs, objective):
    frames = [np.asarray(raw[r["i"]]) for r in recs]
    Ms = [np.asarray(r["M"]) for r in recs]
    return FaceBatch.build(frames, Ms, objective)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["u3d", "cdup3d", "cdup2d", "uniform"], required=True)
    p.add_argument("--n-fit", type=int, default=500, help="U3D: faces (one per video) in the fitness")
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--steps", type=int, default=2300, help="C-DUP: ~3 upstream epochs of batch 256")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eps", type=float, default=EPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--w-id", type=float, default=1.0, help="weight of the identity term (ablation)")
    p.add_argument("--w-ctx", type=float, default=1.0, help="weight of the context term (ablation)")
    p.add_argument("--u3d-noise", choices=["paper", "repo"], default=u3d_core.NOISE,
                   help="paper = Eq. 1-2 (default); repo = the upstream Rust generator, for comparison")
    p.add_argument("--u3d-omega", type=float, default=u3d_core.OMEGA,
                   help="psolib omega; effective inertia is 1+omega (default -0.27 -> 0.73; repo uses 1.2)")
    a = p.parse_args()

    out = RESULTS / "uap" / "universal" / (a.method + a.tag)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log"), logging.StreamHandler()])
    log = logging.getLogger("deepfake").info
    seed_everything(a.seed)

    target = SimSwapTarget()
    objective = ProtectionObjective(target, w_id=a.w_id, w_ctx=a.w_ctx)
    raw, index = load_train_index()
    videos = sorted({r["video"] for r in index})
    log(f"train faces: {len(index)} frames from {len(videos)} videos (face width >= 40 px, score >= 0.6)")

    # held-out validation faces: one frame per video, videos disjoint from the U3D fitness set
    rng = np.random.RandomState(1234)
    perm = rng.permutation(len(videos))
    val_videos = {videos[i] for i in perm[: a.n_val]}
    val_recs = one_frame_per_video([r for r in index if r["video"] in val_videos], a.n_val, seed=1)
    fit_index = [r for r in index if r["video"] not in val_videos]
    val = batch_from(raw, val_recs, objective)
    t0 = time.time()
    meta = {"method": a.method, "eps": a.eps, "eps_255": a.eps * 255, "seed": a.seed,
            "n_train_frames": len(fit_index), "n_train_videos": len({r['video'] for r in fit_index}),
            "n_val": len(val_recs), "w_id": a.w_id, "w_ctx": a.w_ctx}

    if a.method == "u3d":
        fit = batch_from(raw, one_frame_per_video(fit_index, a.n_fit, seed=a.seed), objective)
        best, fbest, hist = u3d_search(objective, fit, seed=a.seed, eps=a.eps, log=log,
                                       noise=a.u3d_noise, omega=a.u3d_omega)
        clip = u3d_clip(best, a.eps, a.u3d_noise)
        pso_cfg = dict(u3d_core.PSO_DEFAULTS, omega=a.u3d_omega, lb=u3d_core.LB, ub=u3d_core.UB)
        meta.update({"params": dict(zip(u3d_core.PARAM_NAMES, best)), "fit_score": fbest,
                     "n_fit": len(fit), "noise": a.u3d_noise, "pso": pso_cfg,
                     "effective_inertia": 1 + a.u3d_omega, "I": u3d_core.I_SHIFTS})
        dump_json(out / "history.json", hist)
    elif a.method in ("cdup3d", "cdup2d"):
        def sample_batch(n, _rng=np.random.RandomState(a.seed)):
            recs = [fit_index[i] for i in _rng.randint(0, len(fit_index), size=n)]
            return batch_from(raw, recs, objective)

        g, hist = cdup_train(objective, sample_batch, two_d=(a.method == "cdup2d"), steps=a.steps,
                             batch=a.batch, eps=a.eps, seed=a.seed, log=log)
        torch.save(g.state_dict(), out / "generator.pt")
        clip = cdup_clip(g, a.eps, seed=a.seed)
        meta.update({"steps": a.steps, "batch": a.batch, "adam": {"lr": 0.002, "beta1": 0.3,
                     "decay": "0.95 every 2000 steps (staircase)"}})
        dump_json(out / "history.json", hist)
    else:
        clip = uniform_clip(a.eps, seed=a.seed)

    val_score, val_parts = clip_score(objective, val, clip)
    meta.update({"val_score": val_score, "val_s_id": val_parts["s_id"], "val_s_ctx": val_parts["s_ctx"],
                 "seconds": round(time.time() - t0, 1), "clip_shape": list(clip.shape),
                 "clip_linf_255": float(clip.abs().max() * 255)})
    np.save(out / "clip.npy", clip.detach().cpu().numpy().astype(np.float32))
    dump_json(out / "meta.json", meta)
    log(f"[{a.method}] val score {val_score:.4f} (id {val_parts['s_id']:.3f}, ctx {val_parts['s_ctx']:.3f}) "
        f"in {meta['seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
