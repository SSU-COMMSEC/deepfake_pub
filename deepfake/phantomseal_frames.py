"""Step 4b - PhantomSeal applied to every scored frame of each test video (per-frame reference).

PhantomSeal protects one image at a time (src/simswap/defense.py: Defense._perturb_imgs). Here
it is run independently on every frame that evaluate.py scores, so the table can compare one
universal perturbation against protecting each frame separately.

A video frame is not the 224 crop PhantomSeal was written for: the defender owns the frame
pixels and the attacker crops the face with its own alignment. Optimising the crop and pasting it
back loses most of the perturbation when the attacker crops again (two resamplings), so the
objective is optimised on the frame through the defender's crop warp ("frame space"). The loss
terms, weights, sign-gradient step, iteration count and best-iterate selection are PhantomSeal's,
unchanged.

Per video:
  * one decoy (cloak) identity for the whole video, by PhantomSeal's distance rule applied to
    every face frame: FaceNet distance above cloak_min_distance on all of them, mean distance
    closest to cloak_distance
  * per-channel bound = the benchmark's L_inf budget (10/255) on R, G and B, so the upload budget
    matches the universal rows. --native-limits uses PhantomSeal's R/G/B = 0.075/0.030/0.075
    (evaluate then still clips the upload to the budget it is given).

    python -m deepfake.phantomseal_frames            # all evaluation videos (resumable)
Output: results/deepfake/uap/per_frame/phantomseal/<stem>.npz (protected uint8 BGR frames at
eval_frames) + meta.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from deepfake.common import EPS, crop_grids, uncrop_grids, warp
from deepfake.inject import quantize_protect
from deepfake.paths import CACHE, PHANTOMSEAL, RESULTS
from deepfake.ucf import read_frames


def l2_per_image(x, y):
    return ((x - y) ** 2).view(x.size(0), -1).mean(dim=1)


@torch.no_grad()
def select_decoy(base, clean_crops):
    """One decoy for the video, PhantomSeal's DistanceCloakSelector rule over all face frames."""
    dataset = base.config.third_party.dataset
    if not dataset.cloak_mix:
        return base.cloak.find_best_cloaks(clean_crops[:1]).cpu(), {"mode": "first_frame"}
    candidates = base.cloak.cloak_imgs["mix"]
    embeddings = [e.cuda() for e in base.cloak.cloak_embeddings["mix"]]
    from src.common_utils import cd          # PhantomSeal resolves data/ relative to its own root
    with cd(PHANTOMSEAL):
        paths = base.cloak._get_cloak_imgs_path()["mix"]
    assert len(paths) == len(candidates), (len(paths), len(candidates))
    crops = clean_crops.permute(0, 2, 3, 1).cpu().numpy() * 255.0
    dist = np.array([base.effectiveness.get_image_distance(c, embeddings) for c in crops], dtype=float)
    dist = dist[~np.isnan(dist).any(axis=1)]
    valid = (dist > dataset.cloak_min_distance).all(axis=0) if dist.size else np.zeros(len(paths), bool)
    if not valid.any():
        return base.cloak.find_best_cloaks(clean_crops[:1]).cpu(), {"mode": "first_frame"}
    best = int(np.argmin(np.where(valid, np.abs(dist.mean(axis=0) - dataset.cloak_distance), np.inf)))
    return candidates[best:best + 1].cpu(), {"mode": "video", "path": Path(paths[best]).name,
                                             "mean_distance": float(dist[:, best].mean())}


def perturb_frames(base, frames, M, cloak_imgs, limits, epochs=None):
    """Defense._perturb_imgs with the optimised variable being the frame (frames [B,3,H,W] in [0,1],
    M [B,2,3] defender alignment). Every term is evaluated on the crop the warp extracts."""
    d = base.config.third_party.defense
    epochs = int(d.epochs if epochs is None else epochs)
    hw = tuple(frames.shape[-2:])
    cgrid = crop_grids(M, hw)
    footprint = (warp(torch.ones(len(frames), 1, 224, 224, device=frames.device), uncrop_grids(M, hw)) > 0).float()
    x = frames.clone().detach() + torch.randn_like(frames) * 1e-5 * footprint
    crops = warp(frames, cgrid)
    with torch.no_grad():
        self_id = base._get_imgs_identity(crops)
        cloak_id = base._get_imgs_identity(cloak_imgs)
        latent = base.target.netG.encoder(warp(x, cgrid))
    step = d.epsilon * (torch.max(x) - torch.min(x)) / 2
    lim = torch.as_tensor(limits, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    best, best_loss = frames.clone(), torch.full((len(frames),), float("inf"), device=frames.device)
    for _ in range(epochs):
        x = x.clone().detach().requires_grad_(True)
        xc = warp(x, cgrid)
        pert = d.weight.perturb * l2_per_image(xc, crops.detach())
        xid = base._get_imgs_identity(xc)
        ident = -d.weight.identity * torch.clamp(l2_per_image(xid, self_id), 0, d.limit.identity)
        cloak = d.weight.cloak * l2_per_image(xid, cloak_id)
        ctx = -d.weight.context * torch.clamp(l2_per_image(base.target.netG.encoder(xc), latent), 0, d.limit.context)
        loss_i = pert + ident + cloak + ctx
        loss_i.mean().backward()
        g = x.grad.sign().detach() if x.grad is not None else torch.zeros_like(x)
        x = torch.clamp(torch.clamp(x.detach() - step * g, frames - lim, frames + lim), 0, 1)
        li = loss_i.detach()
        imp = li < best_loss
        best_loss[imp], best[imp] = li[imp], x[imp].detach()
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, default=EPS)
    p.add_argument("--native-limits", action="store_true", help="PhantomSeal's R/G/B bounds instead of eps")
    p.add_argument("--epochs", type=int, default=0, help="0 = PhantomSeal's config (1000)")
    p.add_argument("--batch", type=int, default=60, help="frames optimised together (PhantomSeal's batch_size)")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--eval-set", default="eval_set.json")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    from deepfake.model import SimSwapTarget
    target = SimSwapTarget()
    base, d = target.base, target.config.third_party.defense
    limits = [d.limit.R, d.limit.G, d.limit.B] if a.native_limits else [a.eps] * 3
    out = RESULTS / "uap" / "per_frame" / ("phantomseal" + a.tag)
    out.mkdir(parents=True, exist_ok=True)
    entries = json.load(open(RESULTS / a.eval_set))["videos"]
    k, n = map(int, a.shard.split("/"))
    todo = [e for e in entries[k::n] if e["eval_frames"]
            and not (out / f"{Path(e['video']).stem}.npz").exists()]
    print(f"[phantomseal per-frame] {len(todo)} videos, limits {np.round(np.array(limits) * 255, 2)}/255, "
          f"{a.epochs or d.epochs} iterations", flush=True)

    # videos are packed into batches of up to --batch frames; each frame keeps its own decoy
    i = 0
    while i < len(todo):
        group, nfr = [], 0
        while i < len(todo) and (not group or nfr + len(todo[i]["eval_frames"]) <= a.batch):
            group.append(todo[i]); nfr += len(todo[i]["eval_frames"]); i += 1
        t0 = time.time()
        clean, Ms, cloaks, info = [], [], [], []
        for e in group:
            stem = Path(e["video"]).stem
            rec = json.load(open(CACHE / "dets" / f"{stem}.json"))
            frames, _ = read_frames(e["video"], rec["n_window"])
            ts = e["eval_frames"]
            f = [cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB) for t in ts]
            Mv = [np.asarray(rec["dets"][t]["M"]) for t in ts]
            fr = torch.from_numpy(np.stack(f)).cuda().permute(0, 3, 1, 2).float().div(255)
            Mt = torch.as_tensor(np.stack(Mv), dtype=torch.float64, device="cuda")
            decoy, dinfo = select_decoy(base, warp(fr, crop_grids(Mt, fr.shape[-2:])))
            clean.append(fr); Ms.append(Mt); cloaks.append(decoy.cuda().expand(len(ts), -1, -1, -1))
            info.append((e, stem, ts, np.stack(f), dinfo))
        with torch.enable_grad():
            prot = perturb_frames(base, torch.cat(clean), torch.cat(Ms), torch.cat(cloaks), limits,
                                  epochs=a.epochs or None)
        prot = prot.permute(0, 2, 3, 1).cpu().numpy()
        sec = (time.time() - t0) / len(group)
        j = 0
        for e, stem, ts, rgb, dinfo in info:
            q = quantize_protect(rgb, prot[j:j + len(ts)], max(limits))
            j += len(ts)
            np.savez_compressed(out / f"{stem}.npz", ts=np.asarray(ts),
                                frames=np.stack([cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in q]))
            with open(out / "meta.jsonl", "a") as fmeta:
                fmeta.write(json.dumps({"video": e["video"], "n_frames": len(ts), "seconds": round(sec, 1),
                                        "limits_255": [x * 255 for x in limits], "decoy": dinfo}) + "\n")
        print(f"[phantomseal per-frame] {i}/{len(todo)} videos ({nfr} frames) {sec:.0f}s/video", flush=True)
        del clean, Ms, cloaks, prot
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
