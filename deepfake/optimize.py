"""The two optimisers, each used in two settings:

    universal  - optimised once on UCF-101 split-1 TRAIN faces, then only applied   ("one UAP")
    per-video  - optimised on the frames of the very test video it protects          ("per video")

U3D-based : Perlin clip (5 parameters) searched by the upstream PSO; fitness = PhantomSeal score
            averaged over faces x I=5 temporal shifts (U3D's E_tau[...]).
C-DUP-based: 3D generator (or 2D + Tile) trained with Adam; every step rolls the generated clips
            by a random shift (C-DUP's Roll) and maximises the PhantomSeal score.
"""

import time

import numpy as np
import torch

from dfbench import cdup_core
from deepfake import u3d_core
from deepfake.common import EPS, FaceBatch, clip_to_crop, protected_crops, score_clip
from deepfake.inject import NATIVE, T_CLIP


# ------------------------------------------------------------------------------------- U3D
def u3d_clip(params, eps=EPS, noise=u3d_core.NOISE, device="cuda") -> torch.Tensor:
    """[T,1,112,112] Perlin clip (same value on R, G, B as add_perlin_noise_to_frame)."""
    vol = u3d_core.perlin_volume(T_CLIP, NATIVE, NATIVE, params, eps, noise=noise, device=device)
    return vol.float()[:, None]


def u3d_search(objective, fb: FaceBatch, I=u3d_core.I_SHIFTS, seed=0, eps=EPS, log=print,
               noise=u3d_core.NOISE, **pso_kw):
    """PSO over [num_octaves, wl_x, wl_y, wl_t, color_period]. Each face gets I random temporal shifts
    (clip frame tau), drawn once so every particle is scored on the same samples."""
    rng = np.random.RandomState(seed)
    taus = torch.as_tensor(rng.randint(0, T_CLIP, size=(I, len(fb))), device=fb.frames.device)
    history, t0 = [], time.time()
    if len(fb) * I <= 512:
        # small (per-video) sets: score all I x N (face, shift) pairs in one batch - same pairs, fewer launches
        fb_all = fb.subset(torch.arange(len(fb), device=fb.frames.device).repeat(I))
        groups = [(fb_all, taus.reshape(-1))]
    else:
        groups = [(fb, taus[j]) for j in range(I)]

    def fitness(params):
        params = [round(p) if i == 0 else float(p) for i, p in enumerate(params)]
        P = clip_to_crop(u3d_clip(params, eps, noise))
        vals = [score_clip(objective, b, P, tau) for b, tau in groups]
        s = float(np.mean([v[0] for v in vals]))
        history.append({"eval": len(history) + 1, "params": params, "score": s,
                        "s_id": float(np.mean([v[1]["s_id"] for v in vals])),
                        "s_ctx": float(np.mean([v[1]["s_ctx"] for v in vals])),
                        "sec": round(time.time() - t0, 1)})
        if log and len(history) % 50 == 0:
            best = max(h["score"] for h in history)
            log(f"[u3d-pso] eval {len(history)}  score {s:.4f}  best {best:.4f}  {time.time()-t0:.0f}s")
        return -s                     # PSO minimises

    best, fbest = u3d_core.pso(fitness, **pso_kw)
    return best, -fbest, history


# ----------------------------------------------------------------------------------- C-DUP
def cdup_train(objective, sample_batch, two_d=False, steps=2300, batch=32, eps=EPS, seed=0,
               log=print, log_every=100):
    """sample_batch(n) -> FaceBatch (with .ref). Returns the trained generator and its history."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    g = (cdup_core.Generator2D() if two_d else cdup_core.Generator3D()).cuda().train()
    opt, sched = cdup_core.make_optimizer(g)
    history, t0, run = [], time.time(), []
    pos = torch.arange(batch) % T_CLIP                       # sample i sits at clip position i mod 16
    for step in range(1, steps + 1):
        fb = sample_batch(batch)
        z = torch.randn(batch, cdup_core.Z_SIZE, device="cuda")
        clip = eps * g(z)                                    # [B,3,16,112,112]
        clip = cdup_core.roll(clip, rng.randint(0, T_CLIP))  # tf.manip.roll(z_out, shift, axis=1)
        frames = clip[torch.arange(batch), :, pos]           # [B,3,112,112]
        crops = protected_crops(fb, clip_to_crop(frames))
        s, info = objective.score(crops, fb.ref)
        loss = -s.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run.append((float(s.mean()), float(info["s_id"].mean()), float(info["s_ctx"].mean())))
        if step % log_every == 0 or step == steps:
            m = np.mean(run, axis=0)
            history.append({"step": step, "score": m[0], "s_id": m[1], "s_ctx": m[2],
                            "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)})
            run = []
            if log:
                log(f"[cdup{'2d' if two_d else '3d'}] step {step}/{steps} score {m[0]:.4f} "
                    f"(id {m[1]:.3f} ctx {m[2]:.3f}) {time.time()-t0:.0f}s")
    return g, history


def cdup_clip(g, eps=EPS, seed=0) -> torch.Tensor:
    """One perturbation from the trained generator -> [T,3,112,112]."""
    p = cdup_core.draw_perturbation(g, eps, seed=seed)       # [3,16,112,112]
    return p.permute(1, 0, 2, 3).contiguous()


def uniform_clip(eps=EPS, seed=0) -> torch.Tensor:
    """Baseline from the U3D paper's benchmark list: uniform noise in [-eps, eps]."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.rand(T_CLIP, 3, NATIVE, NATIVE, device="cuda", generator=gen) * 2 - 1) * eps


def clip_score(objective, fb: FaceBatch, clip: torch.Tensor, tau: int = 0, positions=None):
    """PhantomSeal score of a fixed clip on a FaceBatch (sample i gets clip frame positions[i])."""
    P = clip_to_crop(clip)
    if positions is None:
        positions = torch.arange(len(fb), device=P.device) % T_CLIP
    idx = (torch.as_tensor(positions, device=P.device) + tau) % T_CLIP
    return score_clip(objective, fb, P, idx)
